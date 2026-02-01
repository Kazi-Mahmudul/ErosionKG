"use client";

import { FileText, ExternalLink, TrendingUp, Network, Database } from "lucide-react";
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";

interface Metadata {
    paper_count: number;
    entity_count: number;
    relationship_count: number;
    dois: Array<{
        doi: string;
        file: string;
        citation: string;
    }>;
}

interface ResearchLibraryProps {
    className?: string;
}

export function ResearchLibrary({ className }: ResearchLibraryProps) {
    const [metadata, setMetadata] = useState<Metadata | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch("/api/metadata")
            .then((res) => res.json())
            .then((data) => {
                setMetadata(data);
                setLoading(false);
            })
            .catch((err) => {
                console.error("Failed to load metadata:", err);
                setLoading(false);
            });
    }, []);

    return (
        <div className={cn("flex flex-col h-full bg-gradient-to-br from-card via-card to-secondary/20 border-r backdrop-blur-sm", className)}>
            {/* Header */}
            <div className="p-6 border-b bg-card/50 backdrop-blur-sm">
                <div className="flex items-center gap-3">
                    <div className="relative">
                        <FileText className="h-6 w-6 text-primary" />
                        <div className="absolute -top-1 -right-1 w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                    </div>
                    <div>
                        <h2 className="text-lg font-bold">Research Library</h2>
                        <p className="text-xs text-muted-foreground">Curated Knowledge Base</p>
                    </div>
                </div>
            </div>

            {loading ? (
                <div className="p-6 space-y-4">
                    {[1, 2, 3].map((i) => (
                        <div key={i} className="h-20 bg-secondary/30 rounded-lg animate-pulse" />
                    ))}
                </div>
            ) : metadata ? (
                <div className="flex-1 overflow-y-auto">
                    {/* Stats Grid */}
                    <div className="p-6 space-y-4 border-b">
                        <div className="grid grid-cols-1 gap-4">
                            <div className="group relative overflow-hidden rounded-xl bg-gradient-to-br from-primary/10 to-primary/5 p-4 hover:shadow-lg transition-all hover:scale-[1.02]">
                                <div className="absolute inset-0 bg-gradient-to-r from-primary/0 via-primary/10 to-primary/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000" />
                                <div className="relative flex items-center justify-between">
                                    <div>
                                        <div className="text-3xl font-bold text-primary">{metadata.paper_count}</div>
                                        <div className="text-xs text-muted-foreground font-medium mt-1">Research Papers</div>
                                    </div>
                                    <Database className="h-8 w-8 text-primary/40" />
                                </div>
                            </div>

                            <div className="group relative overflow-hidden rounded-xl bg-gradient-to-br from-blue-500/10 to-blue-500/5 p-4 hover:shadow-lg transition-all hover:scale-[1.02]">
                                <div className="absolute inset-0 bg-gradient-to-r from-blue-500/0 via-blue-500/10 to-blue-500/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000" />
                                <div className="relative flex items-center justify-between">
                                    <div>
                                        <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">{metadata.entity_count}</div>
                                        <div className="text-xs text-muted-foreground font-medium mt-1">Knowledge Entities</div>
                                    </div>
                                    <Network className="h-8 w-8 text-blue-500/40" />
                                </div>
                            </div>

                            <div className="group relative overflow-hidden rounded-xl bg-gradient-to-br from-purple-500/10 to-purple-500/5 p-4 hover:shadow-lg transition-all hover:scale-[1.02]">
                                <div className="absolute inset-0 bg-gradient-to-r from-purple-500/0 via-purple-500/10 to-purple-500/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000" />
                                <div className="relative flex items-center justify-between">
                                    <div>
                                        <div className="text-3xl font-bold text-purple-600 dark:text-purple-400">{metadata.relationship_count}</div>
                                        <div className="text-xs text-muted-foreground font-medium mt-1">Relationships</div>
                                    </div>
                                    <TrendingUp className="h-8 w-8 text-purple-500/40" />
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Publications List */}
                    <div className="p-6">
                        <div className="flex items-center gap-2 mb-4">
                            <h3 className="text-sm font-semibold">Publications</h3>
                            <div className="h-px flex-1 bg-gradient-to-r from-border to-transparent" />
                        </div>
                        <div className="space-y-3">
                            {metadata.dois.map((item, idx) => (
                                <a
                                    key={idx}
                                    href={item.doi}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="group block p-4 rounded-xl bg-gradient-to-br from-secondary/50 to-secondary/30 hover:from-secondary hover:to-secondary/60 transition-all border border-border/30 hover:border-border hover:shadow-lg hover:scale-[1.02]"
                                >
                                    <div className="flex items-start justify-between gap-3">
                                        <div className="flex-1 min-w-0">
                                            <div className="text-sm font-medium group-hover:text-primary transition-colors line-clamp-2">
                                                {item.citation || item.file}
                                            </div>
                                            <div className="text-xs text-muted-foreground mt-2 truncate flex items-center gap-1">
                                                <ExternalLink className="h-3 w-3" />
                                                <span className="truncate">{item.doi.replace("https://doi.org/", "doi:")}</span>
                                            </div>
                                        </div>
                                        <div className="p-2 rounded-lg bg-primary/10 group-hover:bg-primary/20 transition-colors flex-shrink-0">
                                            <ExternalLink className="h-4 w-4 text-primary" />
                                        </div>
                                    </div>
                                </a>
                            ))}
                        </div>
                    </div>
                </div>
            ) : (
                <div className="p-6 text-center text-sm text-muted-foreground">
                    <Database className="h-12 w-12 mx-auto mb-3 opacity-30" />
                    <p>Failed to load library data</p>
                </div>
            )}
        </div>
    );
}
