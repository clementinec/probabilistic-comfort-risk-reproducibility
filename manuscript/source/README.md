# Clean manuscript source

Build from this directory with:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
```

The package contains only the source files and figure assets used by the clean
manuscript and Supplementary Information.
