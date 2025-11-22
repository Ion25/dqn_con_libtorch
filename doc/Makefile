# Makefile para compilar el documento LaTeX del proyecto DQN

# Variables
MAIN = proyecto_dqn
TEX_FILES = $(MAIN).tex
PDF = $(MAIN).pdf

# Compilador LaTeX
LATEX = pdflatex
BIBTEX = bibtex

# Flags
LATEX_FLAGS = -interaction=nonstopmode -halt-on-error

.PHONY: all clean view help

# Target por defecto
all: $(PDF)

# Compilar el documento (dos pasadas para referencias)
$(PDF): $(TEX_FILES)
	@echo "Compilando documento LaTeX..."
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@echo "Segunda pasada para referencias..."
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex > /dev/null 2>&1 || true
	@echo "✅ Documento compilado: $(PDF)"

# Limpiar archivos auxiliares
clean:
	@echo "Limpiando archivos auxiliares..."
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.fls *.fdb_latexmk *.synctex.gz *.nav *.snm *.vrb
	@echo "✅ Archivos auxiliares eliminados"

# Limpiar todo (incluido PDF)
cleanall: clean
	rm -f $(PDF)
	@echo "✅ PDF eliminado"

# Ver el PDF (requiere visor)
view: $(PDF)
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(PDF); \
	elif command -v evince > /dev/null; then \
		evince $(PDF) & \
	elif command -v okular > /dev/null; then \
		okular $(PDF) & \
	else \
		echo "No se encontró visor de PDF. Abre manualmente: $(PDF)"; \
	fi

# Ayuda
help:
	@echo "Uso del Makefile:"
	@echo "  make         - Compilar el documento PDF"
	@echo "  make clean   - Eliminar archivos auxiliares"
	@echo "  make cleanall- Eliminar archivos auxiliares y PDF"
	@echo "  make view    - Abrir el PDF con el visor predeterminado"
	@echo "  make help    - Mostrar esta ayuda"
