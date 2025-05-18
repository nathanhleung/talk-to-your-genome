"use client";

import { useRouter } from "next/navigation";
import { useRef } from "react";

export default function Home() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);

  function uploadGenome() {
    // After file is selected, we redirect to the /genome page
    fileInputRef.current?.click();
  }

  return (
    <main className="relative min-h-screen flex flex-col items-center justify-center text-white overflow-hidden">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-indigo-900 via-purple-900 to-blue-900 z-0"></div>

      {/* Background dots */}
      <div className="absolute inset-0 bg-[radial-gradient(#fff3_1px,transparent_1px)] [background-size:16px_16px]"></div>

      {/* Content container */}
      <div className="flex flex-col container mx-auto px-4 md:px-12 py-16 z-10 max-w-6xl flex-1">
        <div className="flex flex-1 flex-col md:flex-row items-center">
          {/* Left side - copy */}
          <div className="w-full md:w-1/2 text-center md:text-left mb-8 md:mb-0 md:pr-8">
            <h1 className="text-5xl md:text-6xl font-bold mb-6 tracking-tight">
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-teal-300">
                Talk
              </span>{" "}
              to Your Genome
            </h1>

            <p className="text-xl md:text-2xl mb-6 font-light">
              Understand your DNA with the power of conversation. Ask questions,
              get personalized insights.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center md:justify-start">
              <input
                type="file"
                ref={fileInputRef}
                className="hidden"
                onChange={() => {
                  router.push("/genome");
                }}
              />
              <button
                onClick={uploadGenome}
                className="flex gap-2 items-center bg-gradient-to-r from-cyan-600 to-teal-400 text-white font-bold py-4 px-6 rounded-2xl text-lg shadow-2xl hover:scale-105 active:scale-95 active:shadow-sm transition-transform transition-shadow duration-300 ease-out cursor-pointer"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="w-5 h-5"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2m-4-4l-4-4m0 0l-4 4m4-4v12"
                  />
                </svg>
                Upload Genome
              </button>
            </div>
          </div>

          {/* Right side - illustration */}
          <div className="w-full md:w-1/2 flex justify-center md:justify-end">
            <div className="relative w-full max-w-md h-64 md:h-96">
              <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/30 to-teal-500/30 rounded-2xl backdrop-blur-xl p-8 flex flex-col justify-center shadow-lg">
                <div className="flex items-center space-x-3 mb-4">
                  <div className="w-3 h-3 bg-cyan-400 rounded-full"></div>
                  <p className="font-mono text-sm opacity-80">Genome Chat</p>
                </div>

                <div className="bg-white/10 rounded-lg p-3 mb-3 max-w-xs ml-auto">
                  <p className="text-sm">
                    What medications should I avoid with my genetics?
                  </p>
                </div>

                <div className="bg-white/20 rounded-lg p-3 mb-3 max-w-xs">
                  <p className="text-sm">
                    Based on your CYP2D6 variants, consider alternatives to
                    codeine and tramadol.
                  </p>
                </div>

                <div className="bg-white/10 rounded-lg p-3 max-w-xs ml-auto">
                  <p className="text-sm">What about my cardiovascular risk?</p>
                </div>

                <div className="flex items-center mt-4">
                  <div
                    onClick={uploadGenome}
                    className="flex-1 h-8 rounded-lg bg-white/5 flex items-center px-4 cursor-text"
                  >
                    <p className="text-sm text-white/60">
                      Upload genome to talk...
                    </p>
                  </div>
                  <div
                    onClick={uploadGenome}
                    className="ml-2 w-8 h-8 rounded-lg bg-cyan-500 flex items-center justify-center cursor-pointer hover:bg-cyan-400 transition-colors duration-200"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-4 w-4"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                    >
                      <path
                        fillRule="evenodd"
                        d="M10.293 5.293a1 1 0 011.414 0l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414-1.414L12.586 11H5a1 1 0 110-2h7.586l-2.293-2.293a1 1 0 010-1.414z"
                        clipRule="evenodd"
                      />
                    </svg>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div>
          <p className="text-center opacity-80 text-xs mt-8">
            &copy; 2025 Team 12. Powered by{" "}
            <a
              href="https://anthropic.com/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-cyan-400 hover:underline"
            >
              Anthropic
            </a>{" "}
            and{" "}
            <a
              href="https://pharmcat.org/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-cyan-400 hover:underline"
            >
              PharmCAT
            </a>
            .
          </p>
        </div>
      </div>
    </main>
  );
}
