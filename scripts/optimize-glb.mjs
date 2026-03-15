#!/usr/bin/env node
/**
 * Optimize GLB textures to reduce file size for GitHub.
 *
 * Usage:
 *   node scripts/optimize-glb.js [input.glb] [output.glb] [--max-size 1024] [--quality 80]
 *
 * Defaults:
 *   input  = Castle.glb
 *   output = Castle.glb (overwrites in-place)
 *   max-size = 1024 (longest edge in pixels; use 2048 for higher quality)
 *   quality  = 80   (WebP quality, 0-100)
 */

import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { textureCompress, resample } from '@gltf-transform/functions';
import sharp from 'sharp';
import path from 'path';
import { existsSync, statSync } from 'fs';

const args = process.argv.slice(2);

function getArg(flag, defaultVal) {
    const i = args.indexOf(flag);
    return i !== -1 ? args[i + 1] : defaultVal;
}

const inputFile  = args.find(a => !a.startsWith('--') && args.indexOf(a) === 0) || 'Castle.glb';
const outputFile = args.find(a => !a.startsWith('--') && args.indexOf(a) === 1) || inputFile;
const maxSize    = parseInt(getArg('--max-size', '1024'), 10);
const quality    = parseInt(getArg('--quality', '80'), 10);

if (!existsSync(inputFile)) {
    console.error(`Error: "${inputFile}" not found.`);
    process.exit(1);
}

const beforeBytes = statSync(inputFile).size;
console.log(`Input:    ${inputFile} (${(beforeBytes / 1024 / 1024).toFixed(2)} MB)`);
console.log(`Settings: max-size=${maxSize}px, quality=${quality}, format=webp`);
console.log('Optimizing...');

const io = new NodeIO().registerExtensions(ALL_EXTENSIONS);
const document = await io.read(inputFile);

// Resize textures so the longest edge is at most maxSize pixels
await document.transform(
    resample(),
    textureCompress({
        encoder: sharp,
        targetFormat: 'webp',
        resize: [maxSize, maxSize],
        quality,
    }),
);

await io.write(outputFile, document);

const afterBytes = statSync(outputFile).size;
const saved = ((1 - afterBytes / beforeBytes) * 100).toFixed(1);
console.log(`Output:   ${outputFile} (${(afterBytes / 1024 / 1024).toFixed(2)} MB)`);
console.log(`Saved:    ${saved}% reduction`);
