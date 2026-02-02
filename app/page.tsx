"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { Menu, Moon, Sun, GripVertical, PanelRightClose, PanelRightOpen } from "lucide-react";
import { useTheme } from "next-themes";
import { ResearchLibrary } from "@/components/research-library";
import { ChatInterface } from "@/components/chat-interface";
import { GraphVisualizer } from "@/components/graph-visualizer";

export default function HomePage() {
    const [graphData, setGraphData] = useState<{ nodes: any[]; edges: any[] } | undefined>();
    const [leftPanelOpen, setLeftPanelOpen] = useState(true);
    const [rightPanelOpen, setRightPanelOpen] = useState(false); // collapsed by default on mobile
    const [leftWidth, setLeftWidth] = useState(25); // percentage
    const [rightWidth, setRightWidth] = useState(25); // percentage
    const [isDraggingLeft, setIsDraggingLeft] = useState(false);
    const [isDraggingRight, setIsDraggingRight] = useState(false);
    const { theme, setTheme } = useTheme();
    const containerRef = useRef<HTMLDivElement>(null);

    const handleMouseDown = (side: 'left' | 'right') => (e: React.MouseEvent) => {
        e.preventDefault();
        if (side === 'left') setIsDraggingLeft(true);
        else setIsDraggingRight(true);
    };

    const handleMouseMove = useCallback((e: MouseEvent) => {
        if (!containerRef.current) return;
        const containerRect = containerRef.current.getBoundingClientRect();
        const percentage = ((e.clientX - containerRect.left) / containerRect.width) * 100;

        if (isDraggingLeft) {
            setLeftWidth(Math.min(Math.max(percentage, 15), 40));
        } else if (isDraggingRight) {
            const rightPercentage = 100 - percentage;
            setRightWidth(Math.min(Math.max(rightPercentage, 15), 50));
        }
    }, [isDraggingLeft, isDraggingRight]);

    const handleMouseUp = useCallback(() => {
        setIsDraggingLeft(false);
        setIsDraggingRight(false);
    }, []);

    useEffect(() => {
        if (isDraggingLeft || isDraggingRight) {
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', handleMouseUp);
            document.body.style.cursor = 'col-resize';
            document.body.style.userSelect = 'none';
            return () => {
                document.removeEventListener('mousemove', handleMouseMove);
                document.removeEventListener('mouseup', handleMouseUp);
                document.body.style.cursor = '';
                document.body.style.userSelect = '';
            };
        }
    }, [isDraggingLeft, isDraggingRight, handleMouseMove, handleMouseUp]);

    const centerWidth = leftPanelOpen ? 100 - leftWidth - rightWidth : 100 - rightWidth;

    return (
        <div className="h-screen flex flex-col bg-gradient-to-br from-background via-background to-primary/5">
            {/* Premium Header with Gradient & Glow */}
            <header className="border-b bg-card/50 backdrop-blur-xl shadow-lg relative overflow-hidden">
                {/* Animated gradient background */}
                <div className="absolute inset-0 bg-gradient-to-r from-primary/10 via-transparent to-primary/10 animate-gradient-x opacity-50" />

                <div className="relative px-6 py-4 flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <button
                            onClick={() => setLeftPanelOpen(!leftPanelOpen)}
                            className="lg:hidden p-2 hover:bg-secondary rounded-lg transition-all hover:scale-110 active:scale-95"
                            title="Research Library"
                        >
                            <Menu className="h-5 w-5" />
                        </button>

                        {/* Mobile graph panel toggle */}
                        <button
                            onClick={() => setRightPanelOpen(!rightPanelOpen)}
                            className="lg:hidden p-2 hover:bg-secondary rounded-lg transition-all hover:scale-110 active:scale-95"
                            title="Knowledge Graph"
                        >
                            {rightPanelOpen ? (
                                <PanelRightClose className="h-5 w-5" />
                            ) : (
                                <PanelRightOpen className="h-5 w-5" />
                            )}
                        </button>

                        {/* Logo with pulse animation */}
                        <div className="flex items-center gap-3 group">
                            <div className="relative">
                                <div className="absolute inset-0 bg-primary/30 blur-xl rounded-full animate-pulse group-hover:bg-primary/50 transition-all" />
                                <div className="relative bg-gradient-to-br from-primary to-primary/60 text-primary-foreground p-3 rounded-xl shadow-2xl transform group-hover:scale-110 transition-all duration-300">
                                    <svg
                                        className="h-6 w-6"
                                        fill="none"
                                        viewBox="0 0 24 24"
                                        stroke="currentColor"
                                    >
                                        {/* Erosion layers icon */}
                                        <path
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                            strokeWidth={2}
                                            d="M3 17h18M3 12h15M3 7h12M8 17v2m4-2v2m4-2v2"
                                        />
                                    </svg>
                                </div>
                            </div>
                            <div>
                                <h1 className="text-2xl font-bold bg-gradient-to-r from-foreground via-primary to-foreground bg-clip-text text-transparent animate-gradient-x">
                                    ErosionKG
                                </h1>
                                <p className="text-xs text-muted-foreground flex items-center gap-1.5">
                                    <span className="relative flex h-2 w-2">
                                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                                        <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                                    </span>
                                    GraphRAG • Live
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* Theme toggle with premium animation */}
                    <button
                        onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
                        className="p-3 hover:bg-secondary rounded-xl transition-all hover:scale-110 active:scale-95 hover:rotate-180 duration-500"
                    >
                        {theme === "dark" ? (
                            <Sun className="h-5 w-5 text-yellow-500" />
                        ) : (
                            <Moon className="h-5 w-5 text-slate-700" />
                        )}
                    </button>
                </div>
            </header>

            {/* Resizable 3-Panel Layout */}
            <div className="flex-1 overflow-hidden flex" ref={containerRef}>
                {/* Left Panel - Research Library */}
                {leftPanelOpen && (
                    <>
                        <div
                            className="hidden lg:block animate-in fade-in slide-in-from-left-5 duration-500"
                            style={{ width: `${leftWidth}%`, transition: isDraggingLeft ? 'none' : 'width 0.2s' }}
                        >
                            <ResearchLibrary className="h-full" />
                        </div>

                        {/* Left Resize Handle */}
                        <div
                            onMouseDown={handleMouseDown('left')}
                            className="hidden lg:block w-1 bg-border hover:bg-primary/50 transition-colors cursor-col-resize group relative flex items-center justify-center"
                        >
                            <div className="absolute inset-y-0 -left-1 -right-1 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                                <div className="bg-primary/20 backdrop-blur-sm p-1 rounded-full shadow-lg">
                                    <GripVertical className="h-4 w-4 text-primary" />
                                </div>
                            </div>
                        </div>
                    </>
                )}

                {/* Center Panel - Chat Interface */}
                <div
                    className="flex-1 lg:flex-initial animate-in fade-in zoom-in-95 duration-500 center-panel"
                    style={{
                        transition: isDraggingLeft || isDraggingRight ? 'none' : 'width 0.2s'
                    }}
                >
                    <ChatInterface
                        className="h-full"
                        onGraphData={setGraphData}
                    />
                </div>

                {/* Right Resize Handle - desktop only */}
                <div
                    onMouseDown={handleMouseDown('right')}
                    className="hidden lg:flex w-1 bg-border hover:bg-primary/50 transition-colors cursor-col-resize group relative items-center justify-center"
                >
                    <div className="absolute inset-y-0 -left-1 -right-1 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                        <div className="bg-primary/20 backdrop-blur-sm p-1 rounded-full shadow-lg">
                            <GripVertical className="h-4 w-4 text-primary" />
                        </div>
                    </div>
                </div>

                {/* Right Panel - Graph Visualizer (desktop only) */}
                <div
                    className="hidden lg:block animate-in fade-in slide-in-from-right-5 duration-500"
                    style={{ width: `${rightWidth}%`, transition: isDraggingRight ? 'none' : 'width 0.2s' }}
                >
                    <GraphVisualizer
                        className="h-full"
                        graphData={graphData}
                    />
                </div>
            </div>

            {/* Mobile Research Library Overlay */}
            {leftPanelOpen && (
                <div className="lg:hidden fixed inset-0 z-50 bg-black/50 backdrop-blur-sm animate-in fade-in duration-300">
                    <div className="absolute left-0 top-0 bottom-0 w-80 bg-card shadow-2xl animate-in slide-in-from-left-5 duration-300">
                        <div className="p-4 border-b flex items-center justify-between">
                            <h2 className="font-bold">Research Library</h2>
                            <button
                                onClick={() => setLeftPanelOpen(false)}
                                className="p-2 hover:bg-secondary rounded-lg"
                            >
                                ✕
                            </button>
                        </div>
                        <ResearchLibrary className="h-full" />
                    </div>
                </div>
            )}

            {/* Mobile Graph Panel Overlay */}
            {rightPanelOpen && (
                <div className="lg:hidden fixed inset-0 z-50 bg-black/50 backdrop-blur-sm animate-in fade-in duration-300">
                    <div className="absolute right-0 top-0 bottom-0 w-[85%] max-w-md bg-card shadow-2xl animate-in slide-in-from-right duration-300">
                        <div className="p-4 border-b flex items-center justify-between">
                            <h2 className="font-bold">Knowledge Graph</h2>
                            <button
                                onClick={() => setRightPanelOpen(false)}
                                className="p-2 hover:bg-secondary rounded-lg transition-all"
                            >
                                ✕
                            </button>
                        </div>
                        <div className="h-[calc(100%-60px)]">
                            <GraphVisualizer
                                className="h-full"
                                graphData={graphData}
                            />
                        </div>
                    </div>
                </div>
            )}

            <style jsx global>{`
        .center-panel {
          width: 100%;
        }
        @media (min-width: 1024px) {
          .center-panel {
            width: ${centerWidth}%;
          }
        }

        @keyframes gradient-x {
          0%, 100% {
            background-position: 0% 50%;
          }
          50% {
            background-position: 100% 50%;
          }
        }

        .animate-gradient-x {
          background-size: 200% 200%;
          animation: gradient-x 3s ease infinite;
        }

        /* Premium scrollbar */
        ::-webkit-scrollbar {
          width: 8px;
          height: 8px;
        }

        ::-webkit-scrollbar-track {
          background: transparent;
        }

        ::-webkit-scrollbar-thumb {
          background: hsl(var(--primary) / 0.3);
          border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
          background: hsl(var(--primary) / 0.5);
        }
      `}</style>
        </div>
    );
}
