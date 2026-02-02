"use client";

import { Network, Zap, Focus, Download, Filter, Maximize2 } from "lucide-react";
import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import dynamic from "next/dynamic";
import { cn } from "@/lib/utils";
import { useTheme } from "next-themes";

// Dynamically import ForceGraph2D with no SSR
const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), {
    ssr: false,
});

interface GraphVisualizerProps {
    className?: string;
    graphData?: { nodes: any[]; edges: any[] };
    onNodeClick?: (node: string) => void;
}

const MAX_RELEVANT_NODES = 8;

export function GraphVisualizer({
    className,
    graphData,
    onNodeClick,
}: GraphVisualizerProps) {
    const [allData, setAllData] = useState<{ nodes: any[]; links: any[] }>({
        nodes: [],
        links: [],
    });
    const [showAllNodes, setShowAllNodes] = useState(false);
    const [focusMode, setFocusMode] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);
    const [dimensions, setDimensions] = useState({ width: 400, height: 600 });
    const [hoveredLink, setHoveredLink] = useState<any>(null);
    const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
    const fgRef = useRef<any>(null);
    const { resolvedTheme } = useTheme();
    const isDarkMode = resolvedTheme === 'dark';

    // Calculate node degrees and get top N relevant nodes
    const displayData = useMemo(() => {
        if (showAllNodes || allData.nodes.length <= MAX_RELEVANT_NODES) {
            return allData;
        }

        // Calculate degree for each node
        const nodeDegrees: Record<string, number> = {};
        allData.links.forEach((link: any) => {
            const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;
            nodeDegrees[sourceId] = (nodeDegrees[sourceId] || 0) + 1;
            nodeDegrees[targetId] = (nodeDegrees[targetId] || 0) + 1;
        });

        // Sort nodes by degree and take top N
        const sortedNodes = [...allData.nodes].sort((a, b) => {
            return (nodeDegrees[b.id] || 0) - (nodeDegrees[a.id] || 0);
        });
        const topNodes = sortedNodes.slice(0, MAX_RELEVANT_NODES);
        const topNodeIds = new Set(topNodes.map(n => n.id));

        // Filter links to only include those between top nodes
        const filteredLinks = allData.links.filter((link: any) => {
            const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;
            return topNodeIds.has(sourceId) && topNodeIds.has(targetId);
        });

        return { nodes: topNodes, links: filteredLinks };
    }, [allData, showAllNodes]);

    const handleDownload = useCallback(() => {
        if (!fgRef.current || !displayData.nodes.length) return;

        // Zoom to fit with generous padding to ensure all nodes and labels are visible on canvas
        fgRef.current.zoomToFit(400, 120);

        // Wait for zoom animation to finish before capturing
        setTimeout(() => {
            const canvas = containerRef.current?.querySelector('canvas') as HTMLCanvasElement;
            if (!canvas) return;

            // Get canvas pixel data to find actual content bounds
            const ctx = canvas.getContext('2d');
            if (!ctx) return;

            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const pixels = imageData.data;

            // Find bounds of non-transparent/non-white pixels
            let minX = canvas.width, minY = canvas.height, maxX = 0, maxY = 0;

            for (let y = 0; y < canvas.height; y++) {
                for (let x = 0; x < canvas.width; x++) {
                    const idx = (y * canvas.width + x) * 4;
                    const r = pixels[idx];
                    const g = pixels[idx + 1];
                    const b = pixels[idx + 2];
                    const a = pixels[idx + 3];

                    // Check if pixel is not transparent and not pure white
                    if (a > 10 && !(r > 250 && g > 250 && b > 250)) {
                        minX = Math.min(minX, x);
                        minY = Math.min(minY, y);
                        maxX = Math.max(maxX, x);
                        maxY = Math.max(maxY, y);
                    }
                }
            }

            // Add padding around content (like matplotlib tight_layout)
            const padding = 30;
            minX = Math.max(0, minX - padding);
            minY = Math.max(0, minY - padding);
            maxX = Math.min(canvas.width, maxX + padding);
            maxY = Math.min(canvas.height, maxY + padding);

            const cropWidth = maxX - minX;
            const cropHeight = maxY - minY;

            // Use 4x scale for high DPI (300 DPI equivalent for print)
            const scale = 4;
            const exportCanvas = document.createElement('canvas');
            exportCanvas.width = cropWidth * scale;
            exportCanvas.height = cropHeight * scale;

            const exportCtx = exportCanvas.getContext('2d');
            if (exportCtx) {
                // Use smooth scaling for better quality when upscaling
                exportCtx.imageSmoothingEnabled = true;
                exportCtx.imageSmoothingQuality = 'high';

                // Fill with white background
                exportCtx.fillStyle = '#FFFFFF';
                exportCtx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);

                // Draw cropped region scaled up
                exportCtx.drawImage(
                    canvas,
                    minX, minY, cropWidth, cropHeight,
                    0, 0, exportCanvas.width, exportCanvas.height
                );

                // Download as high-quality PNG
                const link = document.createElement('a');
                link.download = `evidence-graph-${new Date().toISOString().slice(0, 10)}.png`;
                link.href = exportCanvas.toDataURL('image/png', 1.0);
                link.click();
            }
        }, 700);
    }, [displayData.nodes.length]);

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

            setAllData({
                nodes: filteredNodes.length > 0 ? filteredNodes : graphData.nodes.slice(0, 15),
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
            <div className="p-6 border-b bg-card/50 backdrop-blur-sm relative">
                <div className="flex items-center gap-3">
                    <div className="relative">
                        <Network className="h-6 w-6 text-primary" />
                        {displayData.nodes.length > 0 && (
                            <Zap className="absolute -top-1 -right-1 h-3 w-3 text-yellow-500 animate-pulse" />
                        )}
                    </div>
                    <div>
                        <h2 className="text-lg font-bold">Evidence Graph</h2>
                        <p className="text-xs text-muted-foreground">
                            {displayData.nodes.length} nodes • {displayData.links.length} connections
                            {!showAllNodes && allData.nodes.length > MAX_RELEVANT_NODES && (
                                <span className="ml-1 text-primary">(filtered)</span>
                            )}
                        </p>
                    </div>
                </div>
                {focusMode && (
                    <div className="mt-2 flex items-center gap-2 text-xs text-primary animate-pulse">
                        <Focus className="h-3 w-3" />
                        <span>Focus mode activated</span>
                    </div>
                )}
                {/* Action buttons */}
                <div className="absolute top-6 right-6 flex items-center gap-2">
                    {allData.nodes.length > MAX_RELEVANT_NODES && (
                        <button
                            onClick={() => setShowAllNodes(!showAllNodes)}
                            className={cn(
                                "p-2 rounded-full transition-colors backdrop-blur-sm border",
                                showAllNodes
                                    ? "bg-primary/20 text-primary border-primary/30"
                                    : "hover:bg-card/80 text-muted-foreground hover:text-primary border-transparent hover:border-border/50"
                            )}
                            title={showAllNodes ? "Show relevant nodes only" : "Show all nodes"}
                        >
                            {showAllNodes ? <Filter className="h-5 w-5" /> : <Maximize2 className="h-5 w-5" />}
                        </button>
                    )}
                    <button
                        onClick={handleDownload}
                        className="p-2 hover:bg-card/80 rounded-full transition-colors text-muted-foreground hover:text-primary backdrop-blur-sm border border-transparent hover:border-border/50"
                        title="Download Graph Image"
                    >
                        <Download className="h-5 w-5" />
                    </button>
                </div>
            </div>

            {/* Graph Canvas */}
            <div className="flex-1 relative overflow-hidden">
                {displayData.nodes.length > 0 ? (
                    <div className="absolute inset-0 overflow-hidden" style={{ maxWidth: '100%', maxHeight: '100%' }}>
                        <ForceGraph2D
                            ref={fgRef}
                            graphData={displayData}
                            width={Math.min(dimensions.width, containerRef.current?.offsetWidth || dimensions.width)}
                            height={Math.min(dimensions.height, containerRef.current?.offsetHeight || dimensions.height)}
                            nodeLabel="label"
                            nodeAutoColorBy="type"
                            nodeCanvasObject={(node: any, ctx: any, globalScale: number) => {
                                const label = node.label || node.id;
                                const fontSize = 16 / globalScale;
                                const nodeRadius = 10;

                                // Draw node circle (solid, no glow for clarity)
                                ctx.beginPath();
                                ctx.arc(node.x, node.y, nodeRadius, 0, 2 * Math.PI);
                                ctx.fillStyle = node.color || "#4299e1";
                                ctx.fill();

                                // Border
                                ctx.strokeStyle = isDarkMode ? "#ffffff" : "#333333";
                                ctx.lineWidth = 2 / globalScale;
                                ctx.stroke();

                                // Draw label - theme aware
                                ctx.font = `bold ${fontSize}px Inter, sans-serif`;
                                const textWidth = ctx.measureText(label).width;
                                const padding = 4 / globalScale;
                                const labelY = node.y + nodeRadius + padding * 3;

                                if (isDarkMode) {
                                    // Dark mode: white text on dark background
                                    ctx.fillStyle = "rgba(0, 0, 0, 0.85)";
                                    ctx.fillRect(
                                        node.x - textWidth / 2 - padding,
                                        labelY - padding,
                                        textWidth + padding * 2,
                                        fontSize + padding * 2
                                    );
                                    ctx.fillStyle = "#ffffff";
                                } else {
                                    // Light mode: black text, no background (transparent)
                                    ctx.fillStyle = "#000000";
                                }

                                ctx.textAlign = "center";
                                ctx.textBaseline = "top";
                                ctx.fillText(label, node.x, labelY);
                            }}
                            onNodeClick={handleNodeClick}
                            onLinkHover={handleLinkHover}
                            linkDirectionalArrowLength={6}
                            linkDirectionalArrowRelPos={1}
                            linkCurvature={0.25}
                            linkLabel={(link: any) => `${link.label || "relates to"}\nSource: ${link.sourceFile || "Unknown"}`}
                            linkColor={(link: any) => link === hoveredLink ? "rgba(66, 153, 225, 0.8)" : "rgba(100, 100, 100, 0.5)"}
                            linkWidth={(link: any) => link === hoveredLink ? 3 : 2}
                            backgroundColor="transparent"
                            enableNodeDrag={true}
                            enableZoomInteraction={true}
                            cooldownTime={3000}
                            d3AlphaDecay={0.02}
                            d3VelocityDecay={0.25}
                            warmupTicks={100}
                            onEngineStop={() => fgRef.current?.zoomToFit(400, 60)}
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
