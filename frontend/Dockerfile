# Multi-stage build for Next.js production frontend
FROM node:20-alpine AS builder

WORKDIR /app

# Fix #6: Accept build-time env var so Next.js bakes the API URL in at build time.
# Set NEXT_PUBLIC_API_URL in Railway's frontend service as a Build Variable.
ARG NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM node:20-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production
ENV PORT=3000

COPY --from=builder /app/package*.json ./
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public

EXPOSE 3000

CMD ["npm", "start"]
