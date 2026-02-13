BUILD_DIR=./build/
FIG_DIR=./figures/


TEX_FILES = cube.tex cube_RotateForward.tex cube_RotateRight.tex cube_reflectX.tex cube_reflectY.tex cube_reflectZ.tex cube_proj.tex

render_tikz:
	mkdir -p $(BUILD_DIR)
	pdflatex -output-directory=$(BUILD_DIR) cube.tex
	pdflatex -output-directory=$(BUILD_DIR) cube_RotateForward.tex
	pdflatex -output-directory=$(BUILD_DIR) cube_RotateRight.tex
	mv $(BUILD_DIR)/*.pdf $(FIG_DIR)


tiks_to_png:
	for f in $(TEX_FILES); do \
		pdflatex -output-directory=$(BUILD_DIR) $$f; \
		pdftocairo -png -r 300 -transp $(BUILD_DIR)/$$(basename $$f .tex).pdf $(FIG_DIR)/$$(basename $$f .tex).png; \
	done