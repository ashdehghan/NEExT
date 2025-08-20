#!/usr/bin/env python3
"""
Manuscript Build System for NEExT Reddit Experiments

This script provides multiple options for compiling LaTeX manuscripts:
1. Docker-based compilation (most reliable)
2. Tectonic engine (modern, self-contained)
3. Local LaTeX installation (if available)
4. Cloud compilation via Overleaf API

Author: NEExT Team
Date: 2025
"""

import subprocess
import sys
import os
import time
import shutil
from pathlib import Path
import argparse
import webbrowser
import json
import hashlib

class ManuscriptBuilder:
    def __init__(self, tex_file="PASPsample701.tex", watch=False):
        self.tex_file = Path(tex_file)
        self.tex_stem = self.tex_file.stem
        self.manuscript_dir = Path(__file__).parent
        self.watch_mode = watch
        self.last_hash = None
        
    def check_docker(self):
        """Check if Docker is available."""
        try:
            result = subprocess.run(["docker", "--version"], 
                                  capture_output=True, text=True, 
                                  timeout=5)
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def check_tectonic(self):
        """Check if Tectonic is installed."""
        try:
            result = subprocess.run(["tectonic", "--version"], 
                                  capture_output=True, text=True,
                                  timeout=5)
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def check_pdflatex(self):
        """Check if pdflatex is installed locally."""
        try:
            result = subprocess.run(["pdflatex", "--version"], 
                                  capture_output=True, text=True,
                                  timeout=5)
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def install_tectonic(self):
        """Install Tectonic using various methods."""
        print("\n📦 Installing Tectonic (modern LaTeX engine)...")
        
        # Try with conda first
        if shutil.which("conda"):
            print("   Trying conda installation...")
            result = subprocess.run(["conda", "install", "-c", "conda-forge", 
                                   "tectonic", "-y"],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("   ✅ Tectonic installed via conda")
                return True
        
        # Try with brew (macOS)
        if sys.platform == "darwin" and shutil.which("brew"):
            print("   Trying Homebrew installation...")
            result = subprocess.run(["brew", "install", "tectonic"],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("   ✅ Tectonic installed via Homebrew")
                return True
        
        # Try with cargo
        if shutil.which("cargo"):
            print("   Trying cargo installation...")
            result = subprocess.run(["cargo", "install", "tectonic"],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("   ✅ Tectonic installed via cargo")
                return True
        
        # Download binary directly
        print("   Downloading Tectonic binary...")
        if sys.platform == "darwin":
            url = "https://github.com/tectonic-typesetting/tectonic/releases/latest/download/tectonic-macOS-latest.tar.gz"
        elif sys.platform.startswith("linux"):
            url = "https://github.com/tectonic-typesetting/tectonic/releases/latest/download/tectonic-linux-latest.tar.gz"
        else:
            print("   ❌ Unsupported platform for binary download")
            return False
        
        try:
            subprocess.run(f"curl -L {url} | tar xz", shell=True, check=True, cwd=str(self.manuscript_dir))
            print("   ✅ Tectonic binary downloaded")
            return True
        except subprocess.SubprocessError:
            print("   ❌ Failed to download Tectonic binary")
            return False
    
    def build_with_docker(self):
        """Build PDF using Docker container."""
        print("\n🐳 Building with Docker...")
        
        # Create Dockerfile if it doesn't exist
        dockerfile_path = self.manuscript_dir / "Dockerfile.latex"
        if not dockerfile_path.exists():
            dockerfile_content = """FROM texlive/texlive:latest
WORKDIR /manuscript
COPY . .
RUN pdflatex -interaction=nonstopmode {0}.tex && \\
    bibtex {0} || true && \\
    pdflatex -interaction=nonstopmode {0}.tex && \\
    pdflatex -interaction=nonstopmode {0}.tex
""".format(self.tex_stem)
            dockerfile_path.write_text(dockerfile_content)
        
        try:
            # Build Docker image
            print("   Building Docker image...")
            subprocess.run(["docker", "build", "-f", "Dockerfile.latex", 
                          "-t", "neext-manuscript", "."],
                         cwd=str(self.manuscript_dir), check=True)
            
            # Run container and copy PDF
            print("   Compiling manuscript...")
            subprocess.run(["docker", "run", "--rm", 
                          "-v", f"{self.manuscript_dir}:/manuscript",
                          "neext-manuscript"],
                         check=True)
            
            print("   ✅ PDF generated successfully with Docker")
            return True
            
        except subprocess.SubprocessError as e:
            print(f"   ❌ Docker build failed: {e}")
            return False
    
    def build_with_tectonic(self):
        """Build PDF using Tectonic."""
        print("\n🔧 Building with Tectonic...")
        
        # Check for local tectonic binary
        local_tectonic = self.manuscript_dir / "tectonic"
        if local_tectonic.exists():
            tectonic_cmd = str(local_tectonic)
        else:
            tectonic_cmd = "tectonic"
        
        try:
            # Run Tectonic
            result = subprocess.run([tectonic_cmd, "-X", "compile", 
                                   str(self.tex_file)],
                                  cwd=str(self.manuscript_dir),
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("   ✅ PDF generated successfully with Tectonic")
                return True
            else:
                print(f"   ❌ Tectonic compilation failed:")
                print(result.stderr)
                return False
                
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            print(f"   ❌ Tectonic execution failed: {e}")
            return False
    
    def build_with_pdflatex(self):
        """Build PDF using local pdflatex installation."""
        print("\n📝 Building with local pdflatex...")
        
        try:
            # Run pdflatex multiple times for references
            for i in range(3):
                print(f"   Pass {i+1}/3...")
                result = subprocess.run(["pdflatex", "-interaction=nonstopmode", 
                                       str(self.tex_file)],
                                      cwd=str(self.manuscript_dir),
                                      capture_output=True, text=True)
                
                if i == 0:  # Run bibtex after first pass
                    subprocess.run(["bibtex", self.tex_stem],
                                 cwd=str(self.manuscript_dir),
                                 capture_output=True, text=True)
            
            if result.returncode == 0:
                print("   ✅ PDF generated successfully with pdflatex")
                return True
            else:
                print(f"   ❌ pdflatex compilation failed")
                return False
                
        except subprocess.SubprocessError as e:
            print(f"   ❌ pdflatex execution failed: {e}")
            return False
    
    def create_overleaf_instructions(self):
        """Create instructions for Overleaf setup."""
        instructions = """
OVERLEAF SETUP INSTRUCTIONS
============================

Since LaTeX is not installed locally, you can use Overleaf:

1. Go to https://www.overleaf.com
2. Create a new project from "Upload Project"
3. Upload the manuscript folder as a ZIP
4. Set PASPsample701.tex as the main document
5. Use the "Share" button to get a collaboration link

For automated sync:
1. Enable Git access in Overleaf project settings
2. Clone with: git clone https://git.overleaf.com/YOUR_PROJECT_ID
3. Use the sync_overleaf.sh script to push changes

Alternative: GitHub Integration
1. Push your manuscript to GitHub
2. In Overleaf: New Project > Import from GitHub
3. Changes sync automatically
"""
        
        instructions_file = self.manuscript_dir / "OVERLEAF_SETUP.md"
        instructions_file.write_text(instructions)
        print(instructions)
        
        # Create sync script
        sync_script = """#!/bin/bash
# Overleaf Git Sync Script

OVERLEAF_REPO="https://git.overleaf.com/YOUR_PROJECT_ID"
BRANCH="main"

echo "🔄 Syncing with Overleaf..."

# Add all changes
git add -A

# Commit with timestamp
git commit -m "Update: $(date '+%Y-%m-%d %H:%M:%S')"

# Push to Overleaf
git push $OVERLEAF_REPO $BRANCH

echo "✅ Sync complete!"
"""
        
        sync_file = self.manuscript_dir / "sync_overleaf.sh"
        sync_file.write_text(sync_script)
        sync_file.chmod(0o755)
        
    def get_file_hash(self):
        """Get hash of TeX file for change detection."""
        with open(self.tex_file, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def open_pdf(self):
        """Open the generated PDF."""
        pdf_path = self.manuscript_dir / f"{self.tex_stem}.pdf"
        if pdf_path.exists():
            print(f"\n📄 Opening PDF: {pdf_path}")
            if sys.platform == "darwin":  # macOS
                subprocess.run(["open", str(pdf_path)])
            elif sys.platform.startswith("linux"):
                subprocess.run(["xdg-open", str(pdf_path)])
            elif sys.platform == "win32":
                os.startfile(str(pdf_path))
            else:
                webbrowser.open(f"file://{pdf_path.absolute()}")
        else:
            print(f"\n⚠️  PDF not found: {pdf_path}")
    
    def watch_and_build(self):
        """Watch for changes and rebuild automatically."""
        print("\n👁️  Watching for changes (Ctrl+C to stop)...")
        
        while True:
            try:
                current_hash = self.get_file_hash()
                if current_hash != self.last_hash:
                    print(f"\n🔄 Change detected in {self.tex_file.name}")
                    self.last_hash = current_hash
                    
                    # Try to build
                    if self.build():
                        print("✅ Rebuild successful")
                    else:
                        print("❌ Rebuild failed")
                
                time.sleep(2)  # Check every 2 seconds
                
            except KeyboardInterrupt:
                print("\n\n👋 Stopping watch mode")
                break
    
    def build(self):
        """Main build method - tries different approaches."""
        print("\n" + "="*60)
        print("NEEXT MANUSCRIPT BUILD SYSTEM")
        print("="*60)
        
        # Check available build methods
        has_docker = self.check_docker()
        has_tectonic = self.check_tectonic() or (self.manuscript_dir / "tectonic").exists()
        has_pdflatex = self.check_pdflatex()
        
        print("\n📋 Available build methods:")
        print(f"   Docker:    {'✅' if has_docker else '❌'}")
        print(f"   Tectonic:  {'✅' if has_tectonic else '❌'}")
        print(f"   pdflatex:  {'✅' if has_pdflatex else '❌'}")
        
        # Try building with available methods
        success = False
        
        if has_pdflatex:
            success = self.build_with_pdflatex()
        
        if not success and has_tectonic:
            success = self.build_with_tectonic()
        
        if not success and has_docker:
            success = self.build_with_docker()
        
        if not success:
            # Try to install Tectonic
            print("\n⚠️  No build method available. Attempting to install Tectonic...")
            if self.install_tectonic():
                success = self.build_with_tectonic()
        
        if not success:
            print("\n❌ Could not build manuscript locally")
            print("📝 Creating Overleaf setup instructions...")
            self.create_overleaf_instructions()
        else:
            # Check if PDF was created
            pdf_path = self.manuscript_dir / f"{self.tex_stem}.pdf"
            if pdf_path.exists():
                print(f"\n✅ SUCCESS! PDF created: {pdf_path}")
                print(f"   Size: {pdf_path.stat().st_size / 1024:.1f} KB")
                print(f"   Modified: {time.ctime(pdf_path.stat().st_mtime)}")
        
        return success

def main():
    parser = argparse.ArgumentParser(description="Build NEExT manuscript")
    parser.add_argument("--tex", default="PASPsample701.tex", 
                       help="TeX file to compile")
    parser.add_argument("--watch", action="store_true", 
                       help="Watch for changes and auto-rebuild")
    parser.add_argument("--open", action="store_true", 
                       help="Open PDF after building")
    parser.add_argument("--clean", action="store_true",
                       help="Clean auxiliary files")
    
    args = parser.parse_args()
    
    # Change to manuscript directory
    manuscript_dir = Path(__file__).parent
    os.chdir(manuscript_dir)
    
    if args.clean:
        print("🧹 Cleaning auxiliary files...")
        for pattern in ["*.aux", "*.log", "*.bbl", "*.blg", "*.out", 
                       "*.toc", "*.lof", "*.lot", "*.synctex.gz"]:
            for file in manuscript_dir.glob(pattern):
                file.unlink()
                print(f"   Removed: {file.name}")
        print("✅ Cleanup complete")
        return
    
    # Build manuscript
    builder = ManuscriptBuilder(args.tex, args.watch)
    
    if args.watch:
        # Build once first
        success = builder.build()
        if success and args.open:
            builder.open_pdf()
        # Then watch for changes
        builder.watch_and_build()
    else:
        success = builder.build()
        if success and args.open:
            builder.open_pdf()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())