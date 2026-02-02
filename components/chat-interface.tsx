"use client";

import { Send, Copy, Check, Download, FileText, TrendingUp, MessageSquare, Plus, Lightbulb } from "lucide-react";
import { useState, useRef, useEffect } from "react";
import { cn } from "@/lib/utils";
import ReactMarkdown from "react-markdown";

interface Message {
    role: "user" | "assistant";
    content: string;
    citations?: Citation[];
    metrics?: Metric[];
    relatedQueries?: string[];
}

interface Citation {
    number: number;
    doi: string;
    page: string;
    source: string;
}

interface Metric {
    value: number;
    unit: string;
    context: string;
}

interface ChatInterfaceProps {
    className?: string;
    onGraphData?: (data: { nodes: any[]; edges: any[] }) => void;
}

// Extract metrics from text (e.g., "15%", "30 t/ha/year")
function extractMetrics(text: string): Metric[] {
    const metrics: Metric[] = [];

    // Match patterns like "15%", "30 t/ha", "0.5 mm/year"
    const patterns = [
        /(\d+(?:\.\d+)?)\s*%/g,
        /(\d+(?:\.\d+)?)\s*(t\/ha|tha|tons?\/hectare)/gi,
        /(\d+(?:\.\d+)?)\s*(mm\/year|cm\/year|m\/year)/gi,
    ];

    patterns.forEach(pattern => {
        let match;
        while ((match = pattern.exec(text)) !== null) {
            metrics.push({
                value: parseFloat(match[1]),
                unit: match[2] || "%",
                context: text.substring(Math.max(0, match.index - 50), Math.min(text.length, match.index + 100)),
            });
        }
    });

    return metrics.slice(0, 3); // Limit to top 3 metrics
}

// Parse citations from text and replace with numbered references
function parseCitations(text: string): { text: string; citations: Citation[] } {
    const citations: Citation[] = [];
    const citationMap = new Map<string, number>();
    let citationCounter = 1;

    // Match pattern: (Source: X, Page: Y | DOI: Z) or variants
    // Making "Page:" optional to handle LLM variations
    const pattern = /\(Source:\s*([^,]+),\s*(?:Page:\s*)?([^|]+)\s*\|\s*DOI:\s*([^)]+)\)/g;

    const processedText = text.replace(pattern, (match, source, page, doi) => {
        const key = `${source.trim()}-${page.trim()}`;

        if (!citationMap.has(key)) {
            citationMap.set(key, citationCounter);
            citations.push({
                number: citationCounter,
                source: source.trim(),
                page: page.trim(),
                doi: doi.trim(),
            });
            citationCounter++;
        }

        const num = citationMap.get(key)!;
        return `<cite data-citation="${num}">[${num}]</cite>`;
    });

    return { text: processedText, citations };
}

export function ChatInterface({ className, onGraphData }: ChatInterfaceProps) {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLTextAreaElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    // Listen for node click events from graph
    useEffect(() => {
        const handleFillInput = (e: Event) => {
            const customEvent = e as CustomEvent;
            console.log("Received fill-chat-input event:", customEvent.detail);
            setInput(customEvent.detail);
            inputRef.current?.focus();
        };

        console.log("Adding fill-chat-input event listener");
        window.addEventListener("fill-chat-input", handleFillInput);

        return () => {
            console.log("Removing fill-chat-input event listener");
            window.removeEventListener("fill-chat-input", handleFillInput);
        };
    }, []);

    const copyToClipboard = async (text: string, index: number) => {
        // Strip HTML tags for plain text copy
        const plainText = text.replace(/<[^>]*>/g, "");
        await navigator.clipboard.writeText(plainText);
        setCopiedIndex(index);
        setTimeout(() => setCopiedIndex(null), 2000);
    };

    const exportAsMarkdown = (message: Message) => {
        let markdown = message.content.replace(/<cite[^>]*>\[(\d+)\]<\/cite>/g, "[$1]");

        if (message.citations && message.citations.length > 0) {
            markdown += "\n\n## References\n\n";
            message.citations.forEach(cit => {
                markdown += `[${cit.number}] ${cit.source}, Page ${cit.page} - ${cit.doi}\n`;
            });
        }

        const blob = new Blob([markdown], { type: "text/markdown" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `erosionkg-answer-${Date.now()}.md`;
        a.click();
        URL.revokeObjectURL(url);
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;

        const userMessage = input.trim();
        setInput("");
        setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
        setIsLoading(true);

        // Resize textarea back to default
        if (inputRef.current) {
            inputRef.current.style.height = "auto";
        }

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: userMessage, stream: true }),
            });

            if (!response.body) throw new Error("No response body");

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let assistantMessage = "";
            let buffer = "";

            setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop() || "";

                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        try {
                            const jsonStr = line.slice(6).trim();
                            if (!jsonStr) continue;

                            const data = JSON.parse(jsonStr);

                            if (data.type === "graph" && onGraphData) {
                                console.log("Graph data received:", data.data);
                                onGraphData(data.data);
                            } else if (data.type === "text") {
                                assistantMessage += data.content;
                                setMessages((prev) => {
                                    const newMessages = [...prev];
                                    newMessages[newMessages.length - 1].content = assistantMessage;
                                    return newMessages;
                                });
                            } else if (data.type === "done") {
                                console.log("Stream completed");
                                // Parse citations and metrics when done
                                const { text, citations } = parseCitations(assistantMessage);
                                const metrics = extractMetrics(assistantMessage);
                                setMessages((prev) => {
                                    const newMessages = [...prev];
                                    newMessages[newMessages.length - 1] = {
                                        ...newMessages[newMessages.length - 1],
                                        content: text,
                                        citations,
                                        metrics,
                                    };
                                    return newMessages;
                                });
                                setIsLoading(false);
                            } else if (data.type === "error") {
                                console.error("Server error:", data.message);
                                throw new Error(data.message);
                            } else if (data.type === "related_queries") {
                                // Add related queries to the last message
                                setMessages((prev) => {
                                    const newMessages = [...prev];
                                    newMessages[newMessages.length - 1] = {
                                        ...newMessages[newMessages.length - 1],
                                        relatedQueries: data.queries,
                                    };
                                    return newMessages;
                                });
                            }
                        } catch (err) {
                            console.warn("Failed to parse SSE data:", line, err);
                        }
                    }
                }
            }

            setIsLoading(false);
        } catch (error) {
            console.error("Chat error:", error);
            setMessages((prev) => [
                ...prev,
                { role: "assistant", content: "Sorry, an error occurred. Please try again." },
            ]);
            setIsLoading(false);
        }
    };

    // Auto-resize textarea
    const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        setInput(e.target.value);
        e.target.style.height = "auto";
        e.target.style.height = Math.min(e.target.scrollHeight, 200) + "px";
    };

    return (
        <div className={cn("flex flex-col h-full bg-gradient-to-b from-background to-secondary/10", className)}>
            {/* Header */}
            <div className="border-b bg-card/50 backdrop-blur-sm px-6 py-4">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <MessageSquare className="h-6 w-6 text-primary" />
                        <div>
                            <h2 className="text-lg font-bold">GraphRAG Assistant</h2>
                            <p className="text-xs text-muted-foreground">Powered by Knowledge Graphs & AI</p>
                        </div>
                    </div>

                    {/* New Chat Button */}
                    {messages.length > 0 && (
                        <button
                            onClick={() => {
                                if (confirm("Start a new chat? This will clear the current conversation.")) {
                                    setMessages([]);
                                    setInput("");
                                }
                            }}
                            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary/10 hover:bg-primary/20 text-primary transition-all hover:scale-105 active:scale-95"
                            title="New Chat"
                        >
                            <Plus className="h-4 w-4" />
                            <span className="text-sm font-medium hidden sm:inline">New Chat</span>
                        </button>
                    )}
                </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
                {messages.length === 0 ? (
                    <div className="text-center mt-16 space-y-6">
                        <div className="inline-flex p-4 rounded-full bg-primary/10 backdrop-blur-sm">
                            <MessageSquare className="h-16 w-16 text-primary" />
                        </div>
                        <div className="space-y-2">
                            <h3 className="text-2xl font-bold">Ask About Erosion Research</h3>
                            <p className="text-muted-foreground max-w-md mx-auto">
                                Explore insights from peer-reviewed research papers using our advanced knowledge graph system
                            </p>
                        </div>
                        <div className="flex flex-wrap gap-2 justify-center max-w-2xl mx-auto">
                            {[
                                "What modulates rill erosion?",
                                "Explain natural vs accelerated erosion",
                                "How does rainfall affect soil loss?",
                            ].map((suggestion, idx) => (
                                <button
                                    key={idx}
                                    onClick={() => setInput(suggestion)}
                                    className="px-4 py-2 rounded-full bg-secondary hover:bg-secondary/80 text-sm transition-all hover:scale-105"
                                >
                                    {suggestion}
                                </button>
                            ))}
                        </div>
                    </div>
                ) : (
                    messages.map((msg, idx) => (
                        <div
                            key={idx}
                            className={cn(
                                "flex animate-in fade-in slide-in-from-bottom-2 duration-500",
                                msg.role === "user" ? "justify-end" : "justify-start"
                            )}
                        >
                            <div
                                className={cn(
                                    "max-w-[85%] rounded-2xl px-6 py-4 shadow-lg relative group",
                                    msg.role === "user"
                                        ? "bg-gradient-to-br from-primary to-primary/80 text-primary-foreground"
                                        : "bg-card border border-border/50 backdrop-blur-sm"
                                )}
                            >
                                {msg.role === "assistant" && (
                                    <div className="absolute top-2 right-2 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                        <button
                                            onClick={() => copyToClipboard(msg.content, idx)}
                                            className="p-2 rounded-lg bg-secondary/80 hover:bg-secondary transition-colors"
                                            title="Copy to clipboard"
                                        >
                                            {copiedIndex === idx ? (
                                                <Check className="h-4 w-4 text-green-500" />
                                            ) : (
                                                <Copy className="h-4 w-4" />
                                            )}
                                        </button>
                                        <button
                                            onClick={() => exportAsMarkdown(msg)}
                                            className="p-2 rounded-lg bg-secondary/80 hover:bg-secondary transition-colors"
                                            title="Export as Markdown"
                                        >
                                            <FileText className="h-4 w-4" />
                                        </button>
                                    </div>
                                )}

                                {msg.role === "assistant" ? (
                                    <div className="prose prose-sm dark:prose-invert max-w-none">
                                        {/* Metrics Dashboard */}
                                        {msg.metrics && msg.metrics.length > 0 && (
                                            <div className="not-prose mb-4 p-4 rounded-lg bg-gradient-to-r from-blue-500/10 to-purple-500/10 border border-border/30">
                                                <h4 className="text-xs font-semibold text-muted-foreground mb-3">Key Metrics</h4>
                                                <div className="flex gap-4">
                                                    {msg.metrics.map((metric, i) => (
                                                        <div key={i} className="flex-1 text-center">
                                                            <div className="text-2xl font-bold text-primary">{metric.value}{metric.unit}</div>
                                                            <div className="text-xs text-muted-foreground mt-1 line-clamp-2">{metric.context}</div>
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        )}

                                        <div
                                            dangerouslySetInnerHTML={{ __html: msg.content }}
                                            onClick={(e) => {
                                                const target = e.target as HTMLElement;
                                                if (target.tagName === "CITE") {
                                                    const citNum = parseInt(target.dataset.citation || "0");
                                                    const citation = msg.citations?.find((c) => c.number === citNum);
                                                    if (citation && citation.doi.startsWith("http")) {
                                                        window.open(citation.doi, "_blank");
                                                    }
                                                }
                                            }}
                                            className="citation-content"
                                        />
                                        {msg.citations && msg.citations.length > 0 && (
                                            <div className="mt-4 pt-4 border-t border-border/30">
                                                <p className="text-xs font-semibold text-muted-foreground mb-2">References:</p>
                                                <div className="space-y-2">
                                                    {msg.citations.map((cit) => (
                                                        <div key={cit.number} className="text-xs text-muted-foreground flex gap-2">
                                                            <span className="font-mono">[{cit.number}]</span>
                                                            <span>
                                                                Page {cit.page} •{" "}
                                                                <a
                                                                    href={cit.doi}
                                                                    target="_blank"
                                                                    rel="noopener noreferrer"
                                                                    className="text-primary hover:underline"
                                                                >
                                                                    View Paper
                                                                </a>
                                                            </span>
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        )}

                                        {/* Related Queries */}
                                        {msg.relatedQueries && msg.relatedQueries.length > 0 && !isLoading && (
                                            <div className="mt-4 pt-4 border-t border-border/30">
                                                <p className="text-xs font-semibold text-muted-foreground mb-3 flex items-center gap-1.5">
                                                    <Lightbulb className="h-3.5 w-3.5" />
                                                    Related Questions
                                                </p>
                                                <div className="flex flex-wrap gap-2">
                                                    {msg.relatedQueries.map((query, i) => (
                                                        <button
                                                            key={i}
                                                            onClick={() => {
                                                                setInput(query);
                                                                inputRef.current?.focus();
                                                            }}
                                                            className="related-query-btn px-3 py-1.5 text-xs rounded-full bg-secondary/80 hover:bg-primary/20 hover:text-primary border border-border/30 hover:border-primary/30 transition-all hover:scale-105 active:scale-95 text-left"
                                                            style={{ animationDelay: `${i * 100}ms` }}
                                                        >
                                                            {query}
                                                        </button>
                                                    ))}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                ) : (
                                    <p className="whitespace-pre-wrap font-medium">{msg.content}</p>
                                )}
                            </div>
                        </div>
                    ))
                )}
                {isLoading && (
                    <div className="flex justify-start animate-in fade-in slide-in-from-bottom-2">
                        <div className="bg-card border rounded-2xl px-6 py-4 shadow-lg">
                            <div className="flex gap-1.5">
                                <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                                <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                                <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                            </div>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Input - ChatGPT Style */}
            <div className="p-6 border-t bg-card/30 backdrop-blur-sm">
                <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
                    <div className="relative flex items-end gap-2 bg-background border border-border/50 rounded-3xl shadow-2xl hover:shadow-primary/5 transition-shadow p-2">
                        <textarea
                            ref={inputRef}
                            value={input}
                            onChange={handleInputChange}
                            onKeyDown={(e) => {
                                if (e.key === "Enter" && !e.shiftKey) {
                                    e.preventDefault();
                                    handleSubmit(e);
                                }
                            }}
                            placeholder="Ask about erosion research..."
                            rows={1}
                            className="flex-1 px-4 py-3 bg-transparent focus:outline-none resize-none max-h-[200px] text-sm"
                            disabled={isLoading}
                        />
                        <button
                            type="submit"
                            disabled={!input.trim() || isLoading}
                            className="p-3 bg-primary text-primary-foreground rounded-full hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-all hover:scale-110 active:scale-95 shadow-lg"
                        >
                            <Send className="h-5 w-5" />
                        </button>
                    </div>
                    <p className="text-xs text-center text-muted-foreground mt-2">
                        Press <kbd className="px-1.5 py-0.5 rounded bg-secondary text-xs">Enter</kbd> to send •{" "}
                        <kbd className="px-1.5 py-0.5 rounded bg-secondary text-xs">Shift + Enter</kbd> for new line
                    </p>
                </form>
            </div>

            <style jsx global>{`
        cite[data-citation] {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          min-width: 24px;
          height: 24px;
          margin: 0 2px;
          padding: 0 6px;
          background: linear-gradient(135deg, hsl(var(--primary)) 0%, hsl(var(--primary) / 0.8) 100%);
          color: hsl(var(--primary-foreground));
          border-radius: 12px;
          font-size: 11px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
          animation: citation-appear 0.3s ease-out;
          box-shadow: 0 2px 8px hsl(var(--primary) / 0.3);
        }

        cite[data-citation]:hover {
          transform: translateY(-2px) scale(1.1);
          box-shadow: 0 4px 12px hsl(var(--primary) / 0.4);
        }

        @keyframes citation-appear {
          from {
            opacity: 0;
            transform: scale(0.8);
          }
          to {
            opacity: 1;
            transform: scale(1);
          }
        }

        .citation-content {
          line-height: 1.7;
        }

        .related-query-btn {
          animation: related-query-appear 0.4s ease-out backwards;
        }

        @keyframes related-query-appear {
          from {
            opacity: 0;
            transform: translateY(10px) scale(0.95);
          }
          to {
            opacity: 1;
            transform: translateY(0) scale(1);
          }
        }

        .related-query-btn:hover {
          box-shadow: 0 2px 12px hsl(var(--primary) / 0.2);
        }
      `}</style>
        </div>
    );
}
