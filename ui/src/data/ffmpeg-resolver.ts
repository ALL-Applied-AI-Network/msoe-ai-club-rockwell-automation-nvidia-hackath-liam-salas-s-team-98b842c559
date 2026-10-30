import ffmpegStatic from "ffmpeg-static";
import { existsSync } from "node:fs";

const SETUP_MESSAGE = "Install UI dev dependencies with `npm ci` in ui/ to provide ffmpeg-static.";

export function resolveFfmpeg(
  environment: NodeJS.ProcessEnv = process.env,
  bundledFfmpeg: string | null | undefined = ffmpegStatic
): string {
  const override = environment.FLUIDMIX_FFMPEG?.trim();
  if (override !== undefined) {
    if (override.length === 0) {
      throw new Error("FLUIDMIX_FFMPEG is set but empty.");
    }
    if (!existsSync(override)) {
      throw new Error(`FLUIDMIX_FFMPEG does not exist: ${override}`);
    }
    return override;
  }
  if (!bundledFfmpeg || !existsSync(bundledFfmpeg)) {
    throw new Error(`ffmpeg-static executable is unavailable. ${SETUP_MESSAGE}`);
  }
  return bundledFfmpeg;
}
