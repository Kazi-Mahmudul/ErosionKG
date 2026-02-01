/** @type {import('next').NextConfig} */
const nextConfig = {
    typescript: {
        ignoreBuildErrors: false,
    },
    eslint: {
        ignoreDuringBuilds: true,
    },
    // Proxy API requests to FastAPI during development
    // Proxy API requests
    async rewrites() {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
        console.log(`Rewriting /api to: ${apiUrl}`);
        return [
            {
                source: '/api/:path*',
                destination: `${apiUrl}/api/:path*`,
            },
        ];
    },
};

export default nextConfig;
