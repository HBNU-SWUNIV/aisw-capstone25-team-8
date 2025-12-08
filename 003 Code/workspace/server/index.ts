// import express from "express";
// import path from "path";
// import fs from "fs";
// import { createServer } from "http";
// import { fileURLToPath } from "url";
// import { spawn } from "child_process";
// import { timeStamp } from "console";

// const __filename = fileURLToPath(import.meta.url);
// const __dirname = path.dirname(__filename);

// async function startServer() {
//   const app = express();
//   const server = createServer(app);

//   // Serve static files from dist/public in production
//   const staticPath =
//     process.env.NODE_ENV === "production"
//       ? path.resolve(__dirname, "public")
//       : path.resolve(__dirname, "..", "dist", "public");

//   const rootPath = path.resolve(__dirname, ".."); 

//   app.use(express.static(staticPath));
//   app.use(express.json());

//   app.post("/api/synthesize", (req, res)=>{
//     const { modelIndex, clothingIndex } = req.body;
//     if (!modelIndex || !clothingIndex) {
//       return res.status(400).json({
//         success: false,
//         error: "modelIndex and clothingIndex are required",
//       });
//     }

//     // 경로 확인
    
//     const personPath = path.join(
//       rootPath,
//       `synthesis_input/model-${modelIndex}.png`
//     );
//     const clothPath = path.join(
//       rootPath,
//       `synthesis_input/clothing-${clothingIndex}.png`
//     );
//     const outputDir = path.join(rootPath, "synthesis_output")

//     // 경로가 없으면 만들기
//     if (!fs.existsSync(outputDir)){
//       fs.mkdirSync(outputDir, {recursive: true});
//     }
    
//     const pythonScript = path.join(__dirname, "..", "synthesis", "pipeline.py")
//     const height = 512;
//     const width = 384;
//     const numInferenceSteps = 15;
//     const guidanceScale = 2.5; 
    
//     const python = spawn("python3",[
//       pythonScript,
//       clothPath,
//       personPath,
//       outputDir,
//       height.toString(),
//       width.toString(),
//       numInferenceSteps.toString(),
//       guidanceScale.toString(),
//     ]);

//     let stdout = "";
//     let stderr = "";
    
//     python.stdout.on("data", (data) => {
//       stdout += data.toString();
//     });

//     python.stderr.on("data", (data) => {
//       stderr += data.toString();
//       console.error("Python stderr:", data.toString());
//     });

//     python.on("close", (code) =>{
//       if (code !== 0){
//         console.error("python script erro:", stderr);
//         return res.status(500).json({
//           success : false,
//           error: `Synthesis failed: ${stderr || "Unknown error"}`,
//         });
//       }

//       try {
//         const lines = stdout.trim().split(/\r?\n/);
//         const jsonLine =
//           [...lines].reverse().find((l) => l.trim().startsWith("{")) ??
//           lines[lines.length - 1];

//         const result = JSON.parse(jsonLine);
//         if (result.success){
//           const relativePath = `/synthesis_output/${result.filename}`;
//           res.json({
//             success:true,
//             imagePath: relativePath,
//             filename: result.filename,
//             message: result.message,
//           });
//         } else {
//           res.status(500).json({
//             success: false,
//             error: result.error,
//           });
//         }
//       } catch(e){
//         console.error("Failed to parse Python output:", stdout);
//         res.status(500).json({
//           success:false,
//           error:"Failed to parse synthesis result",
//         });
//       }
//     });

//     setTimeout(() => {
//       if (python.exitCode === null){
//         python.kill();
//         res.status(504).json({
//           success: false,
//           error: "Synthesis timeout - operation took too long",
//         });
//       }
//     }, 30000000);

//   })

//   app.get("/api/health", (_req, res) => {
//     res.json({status:'ok', timestamp: new Date().toISOString()});
//   });

//   app.use(
//     "/synthesis_output",
//     express.static(path.join(rootPath, "synthesis_output"))
//   );
  
//   const port = process.env.PORT || 3000;

//   server.listen(port, () => {
//     console.log(`Server running on http://localhost:${port}/`);
//     console.log(`CatVTON synthesis API available at /api/synthesize`);
//   });
// }

import express from "express";
import path from "path";
import fs from "fs";
import { createServer } from "http";
import { fileURLToPath } from "url";
import { spawn } from "child_process";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const server = createServer(app);

  // 🔹 CORS 미들웨어 (맨 위에 추가)
  app.use((req, res, next) => {
    res.setHeader("Access-Control-Allow-Origin", "http://localhost:3001");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");
    res.setHeader("Access-Control-Allow-Credentials", "true");
    
    if (req.method === "OPTIONS") {
      return res.sendStatus(200);
    }
    
    next();
  });

  const staticPath =
    process.env.NODE_ENV === "production"
      ? path.resolve(__dirname, "public")
      : path.resolve(__dirname, "..", "dist", "public");

  const rootPath = path.resolve(__dirname, ".."); 

  app.use(express.static(staticPath));
  app.use(express.json());

  // 🔹 SSE 클라이언트 저장소
  const sseClients = new Map<string, express.Response>();

  // 🔹 SSE 엔드포인트
  app.get("/api/synthesis-progress/:sessionId", (req, res) => {
    const { sessionId } = req.params;
    
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.flushHeaders();

    sseClients.set(sessionId, res);

    req.on("close", () => {
      sseClients.delete(sessionId);
    });
  });

  app.post("/api/synthesize", (req, res)=>{
    const { modelIndex, clothingIndex } = req.body;
    if (!modelIndex || !clothingIndex) {
      return res.status(400).json({
        success: false,
        error: "modelIndex and clothingIndex are required",
      });
    }

    const sessionId = `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const personPath = path.join(
      rootPath,
      `synthesis_input/model-${modelIndex}.png`
    );
    const clothPath = path.join(
      rootPath,
      `synthesis_input/clothing-${clothingIndex}.png`
    );
    const outputDir = path.join(rootPath, "synthesis_output")

    if (!fs.existsSync(outputDir)){
      fs.mkdirSync(outputDir, {recursive: true});
    }
    
    // 🔹 즉시 응답 반환
    res.json({
      success: true,
      sessionId: sessionId,
      message: "Synthesis started",
    });
    
    // 🔹 Python 프로세스 실행 (비동기)
    const pythonScript = path.join(__dirname, "..", "synthesis", "pipeline.py")
    const height = 512;
    const width = 384;
    const numInferenceSteps = 15;
    const guidanceScale = 2.5; 
    
    const python = spawn("python3",[
      pythonScript,
      clothPath,
      personPath,
      outputDir,
      height.toString(),
      width.toString(),
      numInferenceSteps.toString(),
      guidanceScale.toString(),
    ]);

    let stdout = "";
    let stderr = "";
    let buffer = ""; // 🔹 청크 버퍼 추가
    
    python.stdout.on("data", (data) => {
      const output = data.toString();
      stdout += output;
      buffer += output; // 🔹 버퍼에 누적
      
      // 🔹 줄 단위로 파싱
      let lines = buffer.split('\n');
      buffer = lines.pop() || ""; // 마지막 불완전한 줄은 버퍼에 유지
      
      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith('{')) {
          try {
            const jsonData = JSON.parse(trimmed);
            if (jsonData.type === 'progress') {
              console.log(`[Progress] Step ${jsonData.step}/${jsonData.total_steps}`);
              
              const client = sseClients.get(sessionId);
              if (client) {
                client.write(`data: ${JSON.stringify(jsonData)}\n\n`);
              }
            }
          } catch (e) {
            // JSON 파싱 실패는 무시
          }
        }
      }
    });

    python.stderr.on("data", (data) => {
      stderr += data.toString();
      console.error("Python stderr:", data.toString());
    });

    python.on("close", (code) =>{
      const client = sseClients.get(sessionId);
      
      if (code !== 0){
        console.error("python script error:", stderr);
        if (client) {
          client.write(`data: ${JSON.stringify({ 
            type: "error", 
            error: `Synthesis failed: ${stderr || "Unknown error"}` 
          })}\n\n`);
          client.end();
          sseClients.delete(sessionId);
        }
        return;
      }

      try {
        const lines = stdout.trim().split(/\r?\n/);
        const jsonLine =
          [...lines].reverse().find((l) => {
            const trimmed = l.trim();
            if (!trimmed.startsWith('{')) return false;
            try {
              const parsed = JSON.parse(trimmed);
              return parsed.success !== undefined && parsed.type !== 'progress';
            } catch {
              return false;
            }
          }) ?? lines[lines.length - 1];

        const result = JSON.parse(jsonLine);
        if (result.success){
          const relativePath = `/synthesis_output/${result.filename}`;
          
          // 🔹 SSE로 최종 결과 전송
          if (client) {
            client.write(`data: ${JSON.stringify({ 
              type: "complete",
              imagePath: relativePath,
              filename: result.filename,
              message: result.message
            })}\n\n`);
            client.end();
            sseClients.delete(sessionId);
          }
        } else {
          if (client) {
            client.write(`data: ${JSON.stringify({ 
              type: "error",
              error: result.error
            })}\n\n`);
            client.end();
            sseClients.delete(sessionId);
          }
        }
      } catch(e){
        console.error("Failed to parse Python output:", stdout);
        if (client) {
          client.write(`data: ${JSON.stringify({ 
            type: "error",
            error: "Failed to parse synthesis result"
          })}\n\n`);
          client.end();
          sseClients.delete(sessionId);
        }
      }
    });

    setTimeout(() => {
      if (python.exitCode === null){
        python.kill();
        const client = sseClients.get(sessionId);
        if (client) {
          client.write(`data: ${JSON.stringify({ 
            type: "error",
            error: "Synthesis timeout"
          })}\n\n`);
          client.end();
          sseClients.delete(sessionId);
        }
      }
    }, 30000000);
  })

  app.get("/api/health", (_req, res) => {
    res.json({status:'ok', timestamp: new Date().toISOString()});
  });

  app.use(
    "/synthesis_output",
    express.static(path.join(rootPath, "synthesis_output"))
  );
  
  const port = process.env.PORT || 3000;

  server.listen(port, () => {
    console.log(`Server running on http://localhost:${port}/`);
    console.log(`CatVTON synthesis API available at /api/synthesize`);
  });
}

startServer().catch(console.error);