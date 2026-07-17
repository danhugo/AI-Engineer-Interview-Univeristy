# Cold Start Latency Budget Breakdown - Q&A

---

## Basics

**Q: What is cold-start latency?**
A: Latency that includes one-time startup and initialization work before steady-state serving.

**Q: What is warm latency?**
A: Latency after the server, model, GPU memory, and common kernels are already initialized.

**Q: Why can the first request be unusually slow?**
A: It may trigger lazy loading, allocations, kernel setup, or graph capture.

---

## Budgeting

**Q: Name common cold-start budget categories.**
A: Process startup, model materialization, GPU setup, kernel setup, warmup, and readiness.

**Q: Why split cold start into segments?**
A: It identifies the dominant delay and the responsible subsystem.

**Q: What timestamp marks endpoint readiness?**
A: The point when the service should safely receive real traffic.

---

## Warmup

**Q: What is model warmup?**
A: Running synthetic inference before real traffic to pre-pay lazy initialization costs.

**Q: Why should warmup shapes match real traffic?**
A: Different prompt lengths or batch sizes may trigger different allocations or kernels.

**Q: Can a basic health check prove the model is warm?**
A: No. It may only prove the process is alive.

---

## Autoscaling

**Q: Why does cold start matter for autoscaling?**
A: New replicas cannot absorb traffic until they finish startup and warmup.

**Q: What mitigates long cold starts?**
A: Warm pools, predictive scaling, preloaded images or weights, and readiness gates.

**Q: Which percentile should be tracked for cold start?**
A: Tail percentiles such as P95 or P99.
