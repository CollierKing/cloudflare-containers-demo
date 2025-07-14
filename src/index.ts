import { Container, loadBalance, getContainer } from "@cloudflare/containers";
import { Hono } from "hono";

// MARK: - Constants
const DEFAULT_ENV_VARS = {
  API_TITLE: "FastAPI Container",
  API_VERSION: "2.0.0",
  ENVIRONMENT: "dev",
};

// Shared container instance name for reuse between scheduled jobs and API calls
const SHARED_CONTAINER_NAME = "worker-container";

// MARK: - Container Class
export class FastAPIContainer extends Container<Env> {
  // Port the container listens on (default: 8080)
  defaultPort = 8080;
  // Time before container sleeps due to inactivity (default: 30s)
  sleepAfter = "5m";
  manualStart = true;

  override onStart() {
    console.log("FastAPI Container successfully started");
  }

  override onStop() {
    console.log("FastAPI Container successfully shut down");
  }

  override onError(error: unknown) {
    console.log("FastAPI Container error:", error);
  }
}

// MARK: - Hono App
// Create Hono app with proper typing for Cloudflare Workers
const app = new Hono<{
  Bindings: {
    FASTAPI_CONTAINER: DurableObjectNamespace<FastAPIContainer>;
    ASSETS: any;
  };
}>();

// MARK: - Endpoints
// Index route - serve static HTML greeting page
app.get("/", (c) => {
  return c.env.ASSETS.fetch(c.req.raw);
});

// Info route with available endpoints (moved from index)
app.get("/info", (c) => {
  return c.json({
    message: "FastAPI Container Worker",
    endpoints: {
      "/": "Welcome page with system overview",
      "/info": "System information and available endpoints",
      "/api/*": "Proxy all requests to FastAPI container (reuses scheduled containers)",
      "/docs": "FastAPI interactive documentation (Swagger UI)",
      "/openapi.json": "OpenAPI schema (JSON format)",
      "/warm/api/*": "Use only pre-warmed containers from scheduled jobs (fails if not running)",
      "/container/:id/api/*": "Route requests to specific container instance",
      "/lb/api/*": "Load balance requests over multiple containers",
      "/singleton/api/*": "Get a single container instance",
      "/health": "Health check endpoint",
      "/container-status": "Check if scheduled containers are running",
    },
    sharedContainer: SHARED_CONTAINER_NAME,
  });
});

// Health check endpoint
app.get("/health", (c) => {
  return c.json({ status: "healthy", timestamp: new Date().toISOString() });
});

// Container status endpoint - check if scheduled containers are running
app.get("/container-status", async (c) => {
  try {
    const container = getContainer(c.env.FASTAPI_CONTAINER, SHARED_CONTAINER_NAME);
    // Try to fetch a simple health check from the container
    const response = await container.fetch(new Request("http://localhost:8080/api/health", { method: "GET" }));
    return c.json({ 
      status: "running", 
      containerName: SHARED_CONTAINER_NAME,
      timestamp: new Date().toISOString(),
      containerResponse: response.status === 200 ? "healthy" : "unhealthy"
    });
  } catch (error) {
    return c.json({ 
      status: "not_running", 
      containerName: SHARED_CONTAINER_NAME,
      timestamp: new Date().toISOString(),
      error: error instanceof Error ? error.message : "Unknown error"
    });
  }
});

// Route to specifically use pre-warmed containers (won't start if not running)
app.all("/warm/api/*", async (c) => {
  console.log("Warm API call triggered at:", new Date().toISOString());
  
  const container = getContainer(c.env.FASTAPI_CONTAINER, SHARED_CONTAINER_NAME);
  
  // Don't call start() - only use if already running from scheduled job
  try {
    return await container.fetch(c.req.raw);
  } catch (error) {
    return c.json({ 
      error: "Container not running. Please wait for scheduled job to start it.",
      timestamp: new Date().toISOString()
    }, 503);
  }
});

// Route requests to a specific container using the container ID
app.all("/container/:id/api/*", async (c) => {
  const id = c.req.param("id");
  const containerId = c.env.FASTAPI_CONTAINER.idFromName(`/container/${id}`);
  const container = c.env.FASTAPI_CONTAINER.get(containerId);

  // Forward the request to the FastAPI container
  const url = new URL(c.req.url);
  // Remove /container/:id prefix from the path
  const apiPath = url.pathname.replace(`/container/${id}`, "");
  url.pathname = apiPath;

  const modifiedRequest = new Request(url.toString(), c.req.raw);
  return await container.fetch(modifiedRequest);
});

// MARK: - Load Balancer
// Load balance requests across multiple containers
app.all("/lb/api/*", async (c) => {
  const container = await loadBalance(c.env.FASTAPI_CONTAINER, 3);

  // Forward the request to the FastAPI container
  const url = new URL(c.req.url);
  // Remove /lb prefix from the path
  const apiPath = url.pathname.replace("/lb", "");
  url.pathname = apiPath;

  const modifiedRequest = new Request(url.toString(), c.req.raw);
  return await container.fetch(modifiedRequest);
});

// Get a single container instance (singleton pattern)
app.all("/singleton/api/*", async (c) => {
  const container = getContainer(c.env.FASTAPI_CONTAINER);

  // Forward the request to the FastAPI container
  const url = new URL(c.req.url);
  // Remove /singleton prefix from the path
  const apiPath = url.pathname.replace("/singleton", "");
  url.pathname = apiPath;

  const modifiedRequest = new Request(url.toString(), c.req.raw);
  return await container.fetch(modifiedRequest);
});

// MARK: - Proxy API Routes
// FastAPI docs endpoint - maps to container's /docs
app.get("/docs", async (c) => {
  console.log("API Docs request triggered at:", new Date().toISOString());

  // Use the same container instance that the scheduled job uses
  const container = getContainer(c.env.FASTAPI_CONTAINER, SHARED_CONTAINER_NAME);
  
  // Start container if not already running
  await container.start({
    envVars: {
      ...DEFAULT_ENV_VARS,
    },
  });
  
  // Create request to FastAPI's docs endpoint
  const url = new URL(c.req.url);
  url.pathname = "/docs";
  const docsRequest = new Request(url.toString(), c.req.raw);
  
  return await container.fetch(docsRequest);
});

// OpenAPI schema endpoint for FastAPI docs
app.get("/openapi.json", async (c) => {
  console.log("OpenAPI schema request triggered at:", new Date().toISOString());

  // Use the same container instance that the scheduled job uses
  const container = getContainer(c.env.FASTAPI_CONTAINER, SHARED_CONTAINER_NAME);
  
  // Start container if not already running
  await container.start({
    envVars: {
      ...DEFAULT_ENV_VARS,
    },
  });
  
  // Create request to FastAPI's openapi.json endpoint
  const url = new URL(c.req.url);
  url.pathname = "/openapi.json";
  const openApiRequest = new Request(url.toString(), c.req.raw);
  
  return await container.fetch(openApiRequest);
});

// Default proxy to FastAPI (routes all /api/* requests to container)
app.all("/api/*", async (c) => {
  console.log("API Call triggered at:", new Date().toISOString());

  // Use the same container instance that the scheduled job uses
  const container = getContainer(c.env.FASTAPI_CONTAINER, SHARED_CONTAINER_NAME);
  
  // Check if container is already running from scheduled job
  // If not, start it with the same configuration as the scheduled job
  await container.start({
    envVars: {
      ...DEFAULT_ENV_VARS,
    },
  });
  
  return await container.fetch(c.req.raw);
});

// Fallback for static assets (serve any other static files)
app.get("/*", (c) => {
  return c.env.ASSETS.fetch(c.req.raw);
});

// MARK: - Worker
export default {
  fetch: app.fetch,
  // Handle scheduled cron jobs
  async scheduled(
    controller: any,
    env: {
      FASTAPI_CONTAINER: DurableObjectNamespace<FastAPIContainer>;
      ASSETS: any;
    },
    ctx: ExecutionContext,
  ) {
    console.log("Cron job triggered at:", new Date().toISOString());

    // Use the same container instance name that API routes use
    await getContainer(env.FASTAPI_CONTAINER, SHARED_CONTAINER_NAME).start({
      entrypoint: ["python", "proc.py"],
      envVars: {
        ...DEFAULT_ENV_VARS,
      },
    });
    console.log("Worker containers started via cron job");
  },
};

