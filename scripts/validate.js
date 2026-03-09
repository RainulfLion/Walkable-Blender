#!/usr/bin/env node
/**
 * Basic validation script for the Castle Walkthrough app.
 * Checks that required files exist and the HTML has expected structure.
 */

const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..');
let passed = 0;
let failed = 0;

function check(description, condition) {
  if (condition) {
    console.log(`  PASS  ${description}`);
    passed++;
  } else {
    console.error(`  FAIL  ${description}`);
    failed++;
  }
}

console.log('Running validation checks...\n');

// Check required files exist
check('index.html exists', fs.existsSync(path.join(root, 'index.html')));
check('Castle.glb exists', fs.existsSync(path.join(root, 'Castle.glb')));

// Check HTML content
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');

check('HTML has DOCTYPE declaration', html.includes('<!DOCTYPE html>'));
check('HTML has Three.js import map', html.includes('"three"'));
check('HTML references Castle.glb', html.includes('Castle.glb'));
check('HTML has start button', html.includes('id="startBtn"'));
check('HTML has canvas setup (WebGLRenderer)', html.includes('WebGLRenderer'));
check('HTML has GLTFLoader import', html.includes('GLTFLoader'));

// Check GLB file is non-empty
const glbStats = fs.statSync(path.join(root, 'Castle.glb'));
check('Castle.glb is non-empty (> 1KB)', glbStats.size > 1024);

console.log(`\nResults: ${passed} passed, ${failed} failed`);

if (failed > 0) {
  process.exit(1);
}
