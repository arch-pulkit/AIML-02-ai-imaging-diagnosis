# Build: pure-Go SQLite driver, so CGO is off and the binary is static.
FROM golang:1.27.3-alpine AS build
ENV CGO_ENABLED=0
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY *.go schema.sql ./
RUN go build -trimpath -ldflags="-s -w" -o /out/api . && mkdir -p /out/data

# Runtime: distroless static (no shell), runs as uid 65532.
FROM gcr.io/distroless/static-debian13:nonroot
COPY --from=build /out/api /api
COPY --from=build --chown=65532:65532 /out/data /data
EXPOSE 8080
ENTRYPOINT ["/api"]