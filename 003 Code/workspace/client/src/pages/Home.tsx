// import { Card, CardContent } from "@/components/ui/card";
// import { Separator } from "@/components/ui/separator";
// import { Sparkles } from "lucide-react";
// import { useState } from "react";
// import { toast } from "sonner";

// export default function Home() {
//   const [selectedModel, setSelectedModel] = useState<number | null>(null);
//   const [selectedClothing, setSelectedClothing] = useState<number | null>(null);
//   const [synthesizedImage, setSynthesizedImage] = useState<string | null>(null);
//   const [isLoading, setIsLoading] = useState(false);

//   const handleGenerateStyle = async () => {
//     if (!selectedModel || !selectedClothing) {
//       toast.error("Please select both a model and clothing item");
//       return;
//     }

//     setIsLoading(true);
//     try {
//       const response = await fetch("/api/synthesize", {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
//         },
//         body: JSON.stringify({
//           modelIndex: selectedModel,
//           clothingIndex: selectedClothing,
//         }),
//       });

//       const data = await response.json();

//       if (data.success) {
//         // 백엔드가 돌고 있는 포트(3000)를 기준으로 전체 URL 만들어주기
//         const backendBaseUrl = "http://localhost:3000";
//         setSynthesizedImage(`${backendBaseUrl}${data.imagePath}`);
//         toast.success("Style generated successfully!");
//       } else {
//         toast.error(data.error || "Failed to generate style");
//       }
//     } catch (error) {
//       console.error("Synthesis error:", error);
//       toast.error("Error generating style");
//     } finally {
//       setIsLoading(false);
//     }
//   };

//   return (
//     <div className="min-h-screen bg-background p-4 md:p-8 flex items-center justify-center">
//       <div className="w-full max-w-[1400px] bg-card rounded-xl shadow-sm border border-border p-6 md:p-10">
        
//         {/* Header Section */}
//         <header className="mb-8 flex justify-between items-end">
//           <div>
//             <h1 className="text-3xl md:text-4xl font-bold text-foreground tracking-tight mb-2">
//               Fashion Gallery
//             </h1>
//             <p className="text-muted-foreground text-lg">
//               Curated Collection 2025
//             </p>
//           </div>
//           <div className="hidden md:block text-right">
//             <p className="text-sm font-medium text-primary">Hanbat National University</p>
//             <p className="text-xs text-muted-foreground">Edition 01</p>
//           </div>
//         </header>

//         <Separator className="mb-8" />

//         {/* Main Layout Grid */}
//         <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 h-full">
          
//           {/* Left Column (Model & Clothing Grids) - Spans 5 columns */}
//           <div className="lg:col-span-5 flex flex-col gap-8">
            
//             {/* Top Grid: Models */}
//             <div className="space-y-4">
//               <div className="flex items-center justify-between">
//                 <h2 className="text-xl font-semibold text-foreground">Models</h2>
//                 <span className="text-xs text-muted-foreground bg-secondary px-2 py-1 rounded-full">4 Items</span>
//               </div>
//               <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
//                 {[1, 2, 3, 4].map((item) => (
//                   <Card 
//                     key={`model-${item}`} 
//                     onClick={() => setSelectedModel(item === selectedModel ? null : item)}
//                     className={`overflow-hidden border-0 shadow-sm transition-all duration-300 group cursor-pointer bg-secondary/30 relative
//                       ${selectedModel === item ? 'ring-2 ring-primary ring-offset-2 shadow-md scale-[1.02]' : 'hover:shadow-md hover:scale-[1.02]'}
//                     `}
//                   >
//                     <CardContent className="p-0 aspect-[3/4] relative">
//                       <img 
//                         src={`/images/model-${item}.png`} 
//                         alt={`Model ${item}`}
//                         className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
//                       />
//                       <div className={`absolute inset-0 transition-colors duration-300 
//                         ${selectedModel === item ? 'bg-primary/10' : 'bg-black/0 group-hover:bg-black/10'}
//                       `} />
//                       {selectedModel === item && (
//                         <div className="absolute top-2 right-2 w-2 h-2 bg-primary rounded-full shadow-sm animate-in fade-in zoom-in duration-200" />
//                       )}
//                     </CardContent>
//                   </Card>
//                 ))}
//               </div>
//             </div>

//             <Separator className="opacity-50" />

//             {/* Middle Grid: Clothing */}
//             <div className="space-y-4">
//               <div className="flex items-center justify-between">
//                 <h2 className="text-xl font-semibold text-foreground">Collection</h2>
//                 <span className="text-xs text-muted-foreground bg-secondary px-2 py-1 rounded-full">4 Items</span>
//               </div>
//               <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
//                 {[1, 2, 3, 4].map((item) => (
//                   <Card 
//                     key={`clothing-${item}`} 
//                     onClick={() => setSelectedClothing(item === selectedClothing ? null : item)}
//                     className={`overflow-hidden border-0 shadow-sm transition-all duration-300 group cursor-pointer bg-secondary/30 relative
//                       ${selectedClothing === item ? 'ring-2 ring-primary ring-offset-2 shadow-md scale-[1.02]' : 'hover:shadow-md hover:scale-[1.02]'}
//                     `}
//                   >
//                     <CardContent className="p-0 aspect-[3/4] relative">
//                       <img 
//                         src={`/images/clothing-${item}.png`} 
//                         alt={`Clothing Item ${item}`}
//                         className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
//                       />
//                       <div className={`absolute inset-0 transition-colors duration-300 
//                         ${selectedClothing === item ? 'bg-primary/10' : 'bg-black/0 group-hover:bg-black/10'}
//                       `} />
//                       {selectedClothing === item && (
//                         <div className="absolute top-2 right-2 w-2 h-2 bg-primary rounded-full shadow-sm animate-in fade-in zoom-in duration-200" />
//                       )}
//                     </CardContent>
//                   </Card>
//                 ))}
//               </div>
//             </div>

//             {/* Bottom Section: Generation Button */}
//             <div className="mt-auto pt-4 flex justify-center">
//               <button 
//                 onClick={handleGenerateStyle}
//                 disabled={isLoading}
//                 className="flex flex-col items-center justify-center gap-3 text-foreground transition-all duration-300 group disabled:opacity-50 disabled:cursor-not-allowed"
//               >
//                 <div className={`p-3 rounded-full bg-background shadow-sm group-hover:scale-110 group-hover:shadow-md transition-all duration-300 ${isLoading ? 'animate-spin' : ''}`}>
//                   <Sparkles className="w-6 h-6 text-primary" />
//                 </div>
//                 <p className="text-sm font-bold">{isLoading ? 'Generating...' : 'Generate New Style'}</p>
//               </button>
//             </div>
//           </div>

//           {/* Right Column (Main Showcase) - Spans 7 columns */}
//           <div className="lg:col-span-7 h-full min-h-[500px] lg:min-h-0">
//             <Card className="h-full overflow-hidden border-0 shadow-md relative group">
//               <CardContent className="p-0 h-full relative">
//                 {synthesizedImage ? (
//                   <img 
//                     src={synthesizedImage} 
//                     alt="Synthesized Style"
//                     className="w-full h-full object-cover"
//                   />
//                 ) : (
//                   <img 
//                     src="/images/main-showcase.png" 
//                     alt="Main Showcase"
//                     className="w-full h-full object-cover"
//                   />
//                 )}
//               </CardContent>
//             </Card>
//           </div>

//         </div>
//       </div>
//     </div>
//   );
// }

import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Sparkles } from "lucide-react";
import { useState, useEffect, useRef } from "react";
import { toast } from "sonner";

export default function Home() {
  const [selectedModel, setSelectedModel] = useState<number | null>(null);
  const [selectedClothing, setSelectedClothing] = useState<number | null>(null);
  const [synthesizedImage, setSynthesizedImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  
  // 🔹 Progress 상태
  const [progressImage, setProgressImage] = useState<string | null>(null);
  const [progressStep, setProgressStep] = useState<number>(0);
  const [totalSteps, setTotalSteps] = useState<number>(0);
  
  const eventSourceRef = useRef<EventSource | null>(null);

  const handleGenerateStyle = async () => {
    if (!selectedModel || !selectedClothing) {
      toast.error("Please select both a model and clothing item");
      return;
    }

    setIsLoading(true);
    setProgressImage(null);
    setProgressStep(0);
    setTotalSteps(0);

    try {
      console.log("🔵 Starting synthesis...");
      
      const response = await fetch("/api/synthesize", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          modelIndex: selectedModel,
          clothingIndex: selectedClothing,
        }),
      });

      const data = await response.json();
      console.log("🔵 Response:", data);

      if (!data.success || !data.sessionId) {
        toast.error(data.error || "Failed to start generation");
        setIsLoading(false);
        return;
      }

      const sessionId = data.sessionId;
      const backendBaseUrl = "http://localhost:3000";

      console.log("🔵 Session ID:", sessionId);
      console.log("🔵 Connecting to SSE...");

      const eventSource = new EventSource(
        `${backendBaseUrl}/api/synthesis-progress/${sessionId}`
      );
      eventSourceRef.current = eventSource;

      eventSource.onopen = () => {
        console.log("✅ SSE connection opened");
      };

      eventSource.onmessage = (event) => {
        console.log("📨 SSE message:", event.data);
        try {
          const progressData = JSON.parse(event.data);

          if (progressData.type === "progress") {
            console.log(
              `🔄 Progress: ${progressData.step}/${progressData.total_steps}`
            );
            setProgressStep(progressData.step);
            setTotalSteps(progressData.total_steps);

            if (progressData.image_path) {
              const imageUrl = `${backendBaseUrl}${progressData.image_path}?t=${Date.now()}`;
              console.log("🖼️ Progress image:", imageUrl);
              setProgressImage(imageUrl);
            }
          } else if (progressData.type === "complete") {
            console.log("✅ Generation complete", progressData);

            // 🔹 최종 이미지 설정
            if (progressData.imagePath) {
              const finalUrl = `${backendBaseUrl}${progressData.imagePath}?t=${Date.now()}`;
              console.log("🎉 Final image:", finalUrl);
              setSynthesizedImage(finalUrl);
              toast.success("Style generated successfully!");
            } else {
              toast.success("Generation finished");
            }

            setIsLoading(false);
            eventSource.close();
            eventSourceRef.current = null;
          } else if (progressData.type === "error") {
            console.error("❌ Backend error:", progressData.error);
            toast.error(progressData.error || "Synthesis failed");
            setIsLoading(false);
            eventSource.close();
            eventSourceRef.current = null;
          }
        } catch (e) {
          console.error("❌ Failed to parse progress data:", e);
        }
      };

      eventSource.onerror = (error) => {
        console.error("❌ SSE error:", error);
        toast.error("Connection lost");
        setIsLoading(false);
        eventSource.close();
        eventSourceRef.current = null;
      };
    } catch (error) {
      console.error("❌ Synthesis error:", error);
      toast.error("Error generating style");
      setIsLoading(false);
    }
  };

  // 🔹 컴포넌트 언마운트 시 SSE 정리
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  return (
    <div className="min-h-screen bg-background p-4 md:p-8 flex items-center justify-center">
      <div className="w-full max-w-[1400px] bg-card rounded-xl shadow-sm border border-border p-6 md:p-10">
        
        {/* Header Section */}
        <header className="mb-8 flex justify-between items-end">
          <div>
            <h1 className="text-3xl md:text-4xl font-bold text-foreground tracking-tight mb-2">
              Fashion Gallery
            </h1>
            <p className="text-muted-foreground text-lg">
              Curated Collection 2025
            </p>
          </div>
          <div className="hidden md:block text-right">
            <p className="text-sm font-medium text-primary">Hanbat National University</p>
            <p className="text-xs text-muted-foreground">Edition 01</p>
          </div>
        </header>

        <Separator className="mb-8" />

        {/* Main Layout Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 h-full">
          
          {/* Left Column (Model & Clothing Grids) - Spans 5 columns */}
          <div className="lg:col-span-5 flex flex-col gap-8">
            
            {/* Top Grid: Models */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-foreground">Models</h2>
                <span className="text-xs text-muted-foreground bg-secondary px-2 py-1 rounded-full">4 Items</span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                {[1, 2, 3, 4].map((item) => (
                  <Card 
                    key={`model-${item}`} 
                    onClick={() => setSelectedModel(item === selectedModel ? null : item)}
                    className={`overflow-hidden border-0 shadow-sm transition-all duration-300 group cursor-pointer bg-secondary/30 relative
                      ${selectedModel === item ? 'ring-2 ring-primary ring-offset-2 shadow-md scale-[1.02]' : 'hover:shadow-md hover:scale-[1.02]'}
                    `}
                  >
                    <CardContent className="p-0 aspect-[3/4] relative">
                      <img 
                        src={`/images/model-${item}.png`} 
                        alt={`Model ${item}`}
                        className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
                      />
                      <div className={`absolute inset-0 transition-colors duration-300 
                        ${selectedModel === item ? 'bg-primary/10' : 'bg-black/0 group-hover:bg-black/10'}
                      `} />
                      {selectedModel === item && (
                        <div className="absolute top-2 right-2 w-2 h-2 bg-primary rounded-full shadow-sm animate-in fade-in zoom-in duration-200" />
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>

            <Separator className="opacity-50" />

            {/* Middle Grid: Clothing */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-foreground">Collection</h2>
                <span className="text-xs text-muted-foreground bg-secondary px-2 py-1 rounded-full">4 Items</span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                {[1, 2, 3, 4].map((item) => (
                  <Card 
                    key={`clothing-${item}`} 
                    onClick={() => setSelectedClothing(item === selectedClothing ? null : item)}
                    className={`overflow-hidden border-0 shadow-sm transition-all duration-300 group cursor-pointer bg-secondary/30 relative
                      ${selectedClothing === item ? 'ring-2 ring-primary ring-offset-2 shadow-md scale-[1.02]' : 'hover:shadow-md hover:scale-[1.02]'}
                    `}
                  >
                    <CardContent className="p-0 aspect-[3/4] relative">
                      <img 
                        src={`/images/clothing-${item}.png`} 
                        alt={`Clothing Item ${item}`}
                        className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
                      />
                      <div className={`absolute inset-0 transition-colors duration-300 
                        ${selectedClothing === item ? 'bg-primary/10' : 'bg-black/0 group-hover:bg-black/10'}
                      `} />
                      {selectedClothing === item && (
                        <div className="absolute top-2 right-2 w-2 h-2 bg-primary rounded-full shadow-sm animate-in fade-in zoom-in duration-200" />
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>

            {/* Bottom Section: Generation Button */}
            <div className="mt-auto pt-4 flex justify-center">
              <button 
                onClick={handleGenerateStyle}
                disabled={isLoading}
                className="flex flex-col items-center justify-center gap-3 text-foreground transition-all duration-300 group disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <div className={`p-3 rounded-full bg-background shadow-sm group-hover:scale-110 group-hover:shadow-md transition-all duration-300 ${isLoading ? 'animate-spin' : ''}`}>
                  <Sparkles className="w-6 h-6 text-primary" />
                </div>
                <p className="text-sm font-bold">
                  {isLoading 
                    ? `Generating... ${progressStep}/${totalSteps}` 
                    : 'Generate New Style'}
                </p>
              </button>
            </div>
          </div>

          {/* Right Column (Main Showcase) - Spans 7 columns */}
          <div className="lg:col-span-7 h-full min-h-[500px] lg:min-h-[700px]">
            <Card className="h-full overflow-hidden border-0 shadow-md relative group">
              <CardContent className="p-0 h-full relative">
                <div className="w-full h-full relative">
                  {/* 🔹 로딩 중이고 progress 이미지가 있으면 표시 */}
                  {isLoading && progressImage ? (
                    <img 
                      key={progressImage}
                      src={progressImage} 
                      className="absolute inset-0 w-full h-full object-contain animate-in fade-in duration-200"
                    />
                  ) : synthesizedImage ? (
                    <img 
                      src={synthesizedImage} 
                      alt="Synthesized Style"
                      className="absolute inset-0 w-full h-full object-contain"
                    />
                  ) : (
                    <img 
                      src="/images/main-showcase.png" 
                      alt="Main Showcase"
                      className="absolute inset-0 w-full h-full object-contain"
                    />
                  )}
                </div>
                
                {/* 🔹 Progress 오버레이 */}
                {isLoading && totalSteps > 0 && (
                  <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 bg-black/70 text-white px-4 py-2 rounded-full text-sm font-medium">
                    Step {progressStep} / {totalSteps}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

        </div>
      </div>
    </div>
  );
}