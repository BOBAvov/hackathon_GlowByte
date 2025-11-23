docker build -t glowbyte .
docker run -p 8080:8080 -v $(pwd)/data:/data glowbyte
