/** Pure geometry and timing helpers for the live video overlay (unit-tested; no DOM). */

export interface Rect { readonly x: number; readonly y: number; readonly width: number; readonly height: number }
export interface Size { readonly width: number; readonly height: number }

/** The rectangle a `object-fit: contain` video occupies inside its container (letterboxing). */
export function containedRect(frame: Size, box: Size): Rect {
  if (frame.width <= 0 || frame.height <= 0 || box.width <= 0 || box.height <= 0) return { x: 0, y: 0, width: 0, height: 0 };
  const scale = Math.min(box.width / frame.width, box.height / frame.height);
  const width = frame.width * scale; const height = frame.height * scale;
  return { x: (box.width - width) / 2, y: (box.height - height) / 2, width, height };
}

/** Map a pixel ROI (xyxy in frame_size pixels) to container pixels, respecting letterboxing. */
export function roiToDisplayRect(roi: readonly [number, number, number, number], frameSize: readonly [number, number], box: Size): Rect {
  const video = containedRect({ width: frameSize[0], height: frameSize[1] }, box);
  if (video.width === 0) return video;
  const sx = video.width / frameSize[0]; const sy = video.height / frameSize[1];
  const [x0, y0, x1, y1] = roi;
  return { x: video.x + Math.min(x0, x1) * sx, y: video.y + Math.min(y0, y1) * sy, width: Math.abs(x1 - x0) * sx, height: Math.abs(y1 - y0) * sy };
}

/** Same mapping for a normalized (0-1) bounding box from the tracker locator. */
export function normalizedToDisplayRect(bbox: readonly [number, number, number, number], frameSize: Size, box: Size): Rect {
  const video = containedRect(frameSize, box);
  const [x0, y0, x1, y1] = bbox;
  return { x: video.x + x0 * video.width, y: video.y + y0 * video.height, width: (x1 - x0) * video.width, height: (y1 - y0) * video.height };
}

export interface VialFrame { readonly globalFrame: number; readonly trayIndex: number; readonly localFrame: number; readonly framesPerVial: number }

/**
 * Cohort videos concatenate the vials. global frame = round(time x fps); tray index = floor(global / framesPerVial);
 * local frame = global mod framesPerVial. If framesPerVial is unknown it is derived from the duration and vial count.
 */
export function frameToVialFrame(currentTime: number, fps: number | null, duration: number, vialCount: number, framesPerVial: number | null): VialFrame {
  const rate = fps && fps > 0 ? fps : 10;
  const globalFrame = Math.max(0, Math.round(currentTime * rate));
  const per = framesPerVial && framesPerVial > 0 ? framesPerVial
    : Math.max(1, Math.round((Number.isFinite(duration) && duration > 0 ? duration * rate : globalFrame + 1) / Math.max(1, vialCount)));
  return { globalFrame, trayIndex: Math.floor(globalFrame / per), localFrame: globalFrame % per, framesPerVial: per };
}
