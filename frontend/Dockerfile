# Use Node LTS image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install

# Copy rest of the app
COPY . .

# Build the app
RUN npm run build

# Serve static files using a simple server
RUN npm install -g serve
CMD ["serve", "-s", "build"]

# Expose the port React uses
EXPOSE 3000
