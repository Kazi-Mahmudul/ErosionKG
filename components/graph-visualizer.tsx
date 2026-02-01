"use client";

import { Network, Zap, Focus } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { cn } from "@/lib/utils";

// Dynamically import ForceGraph2D with no SSR
const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), {
    ssr: false,
});

interface GraphVisualizerProps {
    className?: string;
    graphData?: { nodes: any[]; edges: any[] };
    onNodeClick?: (node: string) => void;
}

export function GraphVisualizer({
    className,
    graphData,
    onNodeClick,
}: GraphVisualizerProps) {
    const [data, setData] = useState<{ nodes: any[]; links: any[] }>({
        nodes: [],
        links: [],
    });
    const [focusMode, setFocusMode] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);
    const [dimensions, setDimensions] = useState({ width: 400, height: 600 });
    const [hoveredLink, setHoveredLink] = useState<any>(null);
    const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

    useEffect(() => {
        if (graphData && graphData.nodes && graphData.edges) {
            console.log("Graph data updated:", graphData);

            // Transform edges to links format and add source metadata
            const links = graphData.edges.map((edge: any) => ({
                source: edge.source,
                target: edge.target,
                label: edge.type,
                sourceFile: edge.sourceFile || "Unknown",
                color: "rgba(100, 100, 100, 0.4)",
            }));

            // Only show nodes that are actually connected (dynamic sub-graph)
            const connectedNodeIds = new Set<string>();
            links.forEach((link: any) => {
                connectedNodeIds.add(link.source);
                connectedNodeIds.add(link.target);
            });

            // Filter nodes to only those in current answer
            const filteredNodes = graphData.nodes.filter((node: any) =>
                connectedNodeIds.has(node.id)
            );

            setData({
                nodes: filteredNodes.length > 0 ? filteredNodes : graphData.nodes.slice(0, 15), // Fallback to first 15 if no connections
                links,
            });
        }
    }, [graphData]);

    useEffect(() => {
        const updateDimensions = () => {
            if (containerRef.current) {
                const rect = containerRef.current.getBoundingClientRect();
                setDimensions({
                    width: Math.floor(rect.width),
                    height: Math.floor(rect.height),
                });
            }
        };

        // Initial update with delay to ensure DOM is ready
        const timer = setTimeout(updateDimensions, 100);
        updateDimensions();

        window.addEventListener("resize", updateDimensions);
        return () => {
            clearTimeout(timer);
            window.removeEventListener("resize", updateDimensions);
        };
    }, []);

    const handleNodeClick = (node: any) => {
        console.log("Node clicked:", node);
        if (node.label) {
            const query = `Tell me more about the relationships of ${node.label}`;
            console.log("Dispatching fill-chat-input event with query:", query);

            // Dispatch custom event for chat interface to listen
            const event = new CustomEvent("fill-chat-input", {
                detail: query,
            });
            window.dispatchEvent(event);

            setFocusMode(true);
            setTimeout(() => setFocusMode(false), 2000);
        }
    };

    const handleLinkHover = (link: any, prevLink: any) => {
        setHoveredLink(link);
    };

    return (
        <div
            className={cn("flex flex-col h-full bg-gradient-to-br from-card via-card to-primary/5 border-l backdrop-blur-sm overflow-hidden", className)}
            ref={containerRef}
        >
            {/* Header */}
            <div className="p-6 border-b bg-card/50 backdrop-blur-sm">
                <div className="flex items-center gap-3">
                    <div className="relative">
                        <Network className="h-6 w-6 text-primary" />
                        {data.nodes.length > 0 && (
                            <Zap className="absolute -top-1 -right-1 h-3 w-3 text-yellow-500 animate-pulse" />
                        )}
                    </div>
                    <div>
                        <h2 className="text-lg font-bold">Evidence Graph</h2>
                        <p className="text-xs text-muted-foreground">
                            {data.nodes.length} nodes • {data.links.length} connections
                        </p>
                    </div>
                </div>
                {focusMode && (
                    <div className="mt-2 flex items-center gap-2 text-xs text-primary animate-pulse">
                        <Focus className="h-3 w-3" />
                        <span>Focus mode activated</span>
                    </div>
                )}
            </div>

            {/* Graph Canvas */}
            <div className="flex-1 relative overflow-hidden">
                {data.nodes.length > 0 ? (
                    <div className="absolute inset-0 overflow-hidden" style={{ maxWidth: '100%', maxHeight: '100%' }}>
                        <ForceGraph2D
                            graphData={data}
                            width={Math.min(dimensions.width, containerRef.current?.offsetWidth || dimensions.width)}
                            height={Math.min(dimensions.height, containerRef.current?.offsetHeight || dimensions.height)}
                            nodeLabel="label"
                            nodeAutoColorBy="type"
                            nodeCanvasObject={(node: any, ctx: any, globalScale: number) => {
                                const label = node.label || node.id;
                                const fontSize = 12 / globalScale;
                                const nodeRadius = 6;

                                // Draw node circle with glow effect
                                ctx.beginPath();
                                ctx.arc(node.x, node.y, nodeRadius, 0, 2 * Math.PI);

                                // Glow
                                ctx.shadowBlur = 10;
                                ctx.shadowColor = node.color || "#4299e1";
                                ctx.fillStyle = node.color || "#4299e1";
                                ctx.fill();
                                ctx.shadowBlur = 0;

                                // Border
                                ctx.strokeStyle = "#fff";
                                ctx.lineWidth = 2 / globalScale;
                                ctx.stroke();

                                // Draw label background
                                ctx.font = `${fontSize}px Inter, sans-serif`;
                                const textWidth = ctx.measureText(label).width;
                                const padding = 4 / globalScale;

                                ctx.fillStyle = "rgba(0, 0, 0, 0.85)";
                                ctx.fillRect(
                                    node.x - textWidth / 2 - padding,
                                    node.y + nodeRadius + padding,
                                    textWidth + padding * 2,
                                    fontSize + padding * 2
                                );

                                // Draw label text
                                ctx.fillStyle = "#ffffff";
                                ctx.textAlign = "center";
                                ctx.textBaseline = "top";
                                ctx.fillText(label, node.x, node.y + nodeRadius + padding * 2);
                            }}
                            onNodeClick={handleNodeClick}
                            onLinkHover={handleLinkHover}
                            linkDirectionalArrowLength={5}
                            linkDirectionalArrowRelPos={1}
                            linkCurvature={0.2}
                            linkLabel={(link: any) => `${link.label || "relates to"}\nSource: ${link.sourceFile || "Unknown"}`}
                            linkColor={(link: any) => link === hoveredLink ? "rgba(66, 153, 225, 0.8)" : "rgba(100, 100, 100, 0.4)"}
                            linkWidth={(link: any) => link === hoveredLink ? 3 : 2}
                            backgroundColor="transparent"
                            enableNodeDrag={true}
                            enableZoomInteraction={true}
                            cooldownTime={3000}
                        />

                        {/* Edge Tooltip */}
                        {hoveredLink && (
                            <div
                                className="absolute pointer-events-none z-50 px-3 py-2 bg-card/95 backdrop-blur-sm border border-border/50 rounded-lg shadow-lg text-xs"
                                style={{
                                    left: tooltipPos.x,
                                    top: tooltipPos.y,
                                }}
                            >
                                <div className="font-semibold text-primary">{hoveredLink.label || "Relationship"}</div>
                                <div className="text-muted-foreground mt-1">Source: {hoveredLink.sourceFile || "Unknown"}</div>
                            </div>
                        )}

                        <div className="absolute bottom-4 right-4 px-3 py-2 bg-card/90 backdrop-blur-sm rounded-lg border border-border/50 text-xs space-y-1">
                            <p className="text-muted-foreground">💡 Click nodes to explore relationships</p>
                            <p className="text-muted-foreground">🔍 Hover edges to see source papers</p>
                        </div>
                    </div>
                ) : (
                    <div className="absolute inset-0 flex items-center justify-center">
                        <div className="text-center space-y-4 max-w-xs">
                            <div className="inline-flex p-6 rounded-full bg-gradient-to-br from-primary/20 to-primary/5 backdrop-blur-sm">
                                <Network className="h-16 w-16 text-primary/50" />
                            </div>
                            <div className="space-y-2">
                                <h3 className="font-semibold text-muted-foreground">No Graph Data Yet</h3>
                                <p className="text-sm text-muted-foreground/70">
                                    Ask a question to see the knowledge graph visualization
                                </p>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
