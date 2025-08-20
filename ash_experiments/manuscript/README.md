# NEExT Reddit Experiments Manuscript

Clean manuscript directory for documenting NEExT framework experiments on Reddit data.

## Directory Structure

```
manuscript/
├── figures/               # Drop PNG/PDF images here
├── neext_reddit.tex       # Main manuscript
├── neext_reddit.bib       # Bibliography
├── neext_reddit.pdf       # Compiled PDF
├── aastex701.cls          # PASP journal class
├── aasjournalv7.bst       # Bibliography style
├── Makefile               # Build commands
└── build_manuscript.py    # Build script
```

## Quick Start

```bash
# Build PDF
make build

# Build and open
make open

# Watch for changes
make watch

# Clean auxiliary files
make clean
```

## Manuscript Structure

The manuscript has the following sections ready to be filled:

1. **Introduction** - Problem statement and contributions
2. **Background** - NEExT framework and Reddit dataset
3. **Related Work** - Prior research
4. **Methods** - Our approach
5. **Experiments** - Experimental setup
6. **Results** - Findings and metrics
7. **Discussion** - Analysis and implications
8. **Conclusion** - Summary and future work

## Adding Figures

1. Drop your PNG/PDF files into the `figures/` directory
2. Include in manuscript:
```latex
\begin{figure}[ht!]
\includegraphics[width=\linewidth]{filename.png}
\caption{Your caption here.}
\label{fig:label}
\end{figure}
```

## Writing Workflow

1. Edit `neext_reddit.tex` as experiments progress
2. Run `make watch` to auto-compile on save
3. Add citations to `neext_reddit.bib` as needed
4. Use `\added{}` for revision tracking

## Notes

- PASP abstract limit: 300 words
- Line numbers enabled for review
- Bibliography uses AAS journal style
- Tectonic installed for compilation