const express = require("express");
const app = express();

// A simple route
app.get("/", (req, res) => {
  res.send("Hello from Nova project!");
});

// Server listens on port 3000
app.listen(3000, () => {
  console.log("🚀 Server is running on http://localhost:3000");
});
