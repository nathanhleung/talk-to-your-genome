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

import { locusAtom } from "@/app/genome/atoms";
// Adds `igv` to the global scope
import "igv";
import { useAtomValue } from "jotai";
import { useEffect, useRef } from "react";

export default function IGV() {
  const locus = useAtomValue(locusAtom);
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

  useEffect(() => {
    if (igvBrowserRef.current) {
      const browser = igvBrowserRef.current;

      // @ts-expect-error no typings
      browser.search(locus);

      const igvDiv = igvDivRef.current;
      if (!igvDiv) {
        return;
      }
    }
  }, [locus]);

  return <div id="igv" ref={igvDivRef} />;
}
