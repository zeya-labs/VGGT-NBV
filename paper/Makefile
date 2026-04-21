TEX ?= $(HOME)/.local/bin/tectonic

.PHONY: all clean

all:
	$(TEX) main.tex --keep-logs --keep-intermediates

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.synctex.gz *.toc
	rm -f main.blg main.bbl main.bcf main.run.xml
