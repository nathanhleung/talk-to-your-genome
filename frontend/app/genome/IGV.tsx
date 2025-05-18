"use client";

declare global {
  interface Window {
    igv: {
      createBrowser: (
        element: HTMLElement,
        options: unknown
      ) => Promise<unknown>;
      removeBrowser: (browser: unknown) => void;
      removeAllBrowsers: () => void;
    };
  }
}

// Adds `igv` to the global scope
import "igv";
import { useEffect, useRef } from "react";

export default function IGV() {
  const igvDivRef = useRef<HTMLDivElement>(null);
  const igvBrowserRef = useRef<unknown>(null);

  useEffect(() => {
    const options = {
      genome: "hg38",
    };

    window.igv.createBrowser(igvDivRef.current!, options).then((browser) => {
      if (igvBrowserRef.current) {
        window.igv.removeBrowser(igvBrowserRef.current);
      }

      igvBrowserRef.current = browser;
    });

    return () => {
      if (igvBrowserRef.current) {
        window.igv.removeBrowser(igvBrowserRef.current);
        igvBrowserRef.current = null;
      }
    };
  }, []);

  return <div id="igv" ref={igvDivRef} />;
}
