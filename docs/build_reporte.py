"""Genera `docs/reporte.pdf` a partir de `docs/reporte.md`.

Uso:
    python docs/build_reporte.py

Por que una cadena propia
-------------------------
La maquina de desarrollo no tiene pandoc, LaTeX ni wkhtmltopdf, asi que el PDF
se arma con lo que si esta disponible:

    Markdown  --(markdown)-->  HTML  --(PyMuPDF Story)-->  PDF

Las formulas matematicas se renderizan a PNG con el motor `mathtext` de
matplotlib y se incrustan como imagenes, porque ni el HTML intermedio ni el
motor de Story saben de LaTeX. Las nueve figuras obligatorias se copian de
`artifacts/figures/` al directorio de armado.

Si una formula no se puede renderizar, se deja como texto monoespaciado y se
reporta al final, para que ninguna falla pase inadvertida.
"""

from __future__ import annotations

import html as html_lib
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymupdf

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import config  # noqa: E402

DIR_ARMADO = RAIZ / "docs" / "_build_reporte"
RUTA_MARKDOWN = RAIZ / "docs" / "reporte.md"
RUTA_PDF = RAIZ / "docs" / "reporte.pdf"

#: Tamano de fuente con que se rasterizan las formulas.
PUNTOS_DISPLAY = 15
PUNTOS_INLINE = 10
DPI_FORMULA = 220

#: Ancho util de la pagina carta con los margenes del reporte, en puntos.
ANCHO_FIGURA = 504

#: Sustituciones para que las formulas sean validas en mathtext, que no soporta
#: \text ni acentos dentro del modo matematico.
SUSTITUCIONES = {
    r"\text{": r"\mathrm{",
    r"\lvert": r"|",
    r"\rvert": r"|",
    # mathtext no conoce \le ni \ge: solo \leq y \geq.
    r"\le ": r"\leq ",
    r"\ge ": r"\geq ",
    r"\;\;": r"\ \ ",
    r"\,": r"\ ",
    r"\;": r"\ ",
}

#: Palabras acentuadas que aparecen dentro de \mathrm{} y se reemplazan por su
#: version sin acento: mathtext no rasteriza acentos en modo matematico.
SIN_ACENTOS = str.maketrans("áéíóúñÁÉÍÓÚÑ", "aeiounAEIOUN")

def _espaciar_texto_literal(formula: str) -> str:
    r"""Preserva los espacios dentro de \mathrm{...}.

    mathtext ignora los espacios normales en modo matematico, asi que
    \mathrm{incertidumbre del modelo} se renderiza como
    "incertidumbredelmodelo". Cada espacio dentro de una llave de \mathrm se
    sustituye por `\ `, que mathtext si respeta.
    """
    resultado = []
    posicion = 0
    while True:
        inicio = formula.find(r"\mathrm{", posicion)
        if inicio == -1:
            resultado.append(formula[posicion:])
            break
        apertura = inicio + len(r"\mathrm{")
        profundidad, cierre = 1, apertura
        while cierre < len(formula) and profundidad:
            if formula[cierre] == "{":
                profundidad += 1
            elif formula[cierre] == "}":
                profundidad -= 1
                if not profundidad:
                    break
            cierre += 1
        # Se normaliza primero: si el autor ya escribio "\ ", volverlo a
        # sustituir produciria "\ " y mathtext fallaria.
        contenido = formula[apertura:cierre].replace(r"\ ", " ").replace(" ", r"\ ")
        resultado.append(formula[posicion:apertura] + contenido)
        posicion = cierre
    return "".join(resultado)


ESTILO = """
<style>
body { font-family: sans-serif; font-size: 9.2pt; line-height: 1.45; }
h1 { font-size: 19pt; margin-bottom: 2pt; }
h2 { font-size: 13.5pt; margin-top: 16pt; margin-bottom: 4pt; }
h3 { font-size: 11pt; margin-top: 12pt; margin-bottom: 3pt; }
h4 { font-size: 9.8pt; margin-top: 9pt; }
p { margin-top: 3pt; margin-bottom: 3pt; }
table { border-collapse: collapse; width: 100%; font-size: 7.6pt; margin: 6pt 0; }
th { background-color: #e8eef5; border: 0.5px solid #9aa7b4; padding: 2.5pt; text-align: left; }
td { border: 0.5px solid #c3ccd5; padding: 2.5pt; }
code { font-family: monospace; font-size: 8.2pt; background-color: #f2f4f7; }
pre { font-family: monospace; font-size: 8pt; background-color: #f2f4f7; padding: 4pt; }
li { margin-top: 1.5pt; margin-bottom: 1.5pt; }
hr { border: 0.5px solid #c3ccd5; margin: 10pt 0; }
.formula { margin: 7pt 0; }
.figura { margin: 10pt 0; page-break-before: always; }
</style>
"""


class RenderizadorFormulas:
    """Rasteriza formulas de LaTeX a PNG con el motor mathtext de matplotlib."""

    def __init__(self, destino: Path) -> None:
        self.destino = destino
        self.destino.mkdir(parents=True, exist_ok=True)
        self.contador = 0
        self.fallidas: list[str] = []
        self.renderizadas = 0

    def _normalizar(self, formula: str) -> str:
        """Adapta la formula a la sintaxis que mathtext entiende."""
        texto = formula.strip()
        for viejo, nuevo in SUSTITUCIONES.items():
            texto = texto.replace(viejo, nuevo)
        texto = _espaciar_texto_literal(texto)
        return texto.translate(SIN_ACENTOS)

    def renderizar(self, formula: str, puntos: int) -> tuple[str, int, int] | None:
        """Rasteriza una formula y devuelve `(nombre, ancho, alto)` en puntos.

        Args:
            formula: cuerpo de la formula, sin los delimitadores de dolar.
            puntos: tamano de fuente.

        Returns:
            Tupla con el nombre del PNG y su tamano, o None si no se pudo
            renderizar.
        """
        texto = self._normalizar(formula)
        self.contador += 1
        nombre = f"formula_{self.contador:04d}.png"

        figura = plt.figure(figsize=(0.01, 0.01))
        try:
            figura.text(0, 0, f"${texto}$", fontsize=puntos)
            figura.savefig(
                self.destino / nombre, dpi=DPI_FORMULA, bbox_inches="tight",
                pad_inches=0.02, transparent=False, facecolor="white",
            )
        except Exception as error:  # mathtext no soporta todo LaTeX
            plt.close(figura)
            self.fallidas.append(f"{formula.strip()[:70]} -> {type(error).__name__}")
            return None
        plt.close(figura)

        # PyMuPDF honra el DPI que matplotlib escribe en el PNG, asi que el
        # `rect` ya viene en PUNTOS y no en pixeles. Volver a escalarlo por
        # 72/DPI seria una doble conversion y dejaria las formulas a un tercio
        # de su tamano (una formula de 10 pt se colocaba a 3 pt).
        with pymupdf.open(self.destino / nombre) as imagen:
            ancho_pt, alto_pt = imagen[0].rect.width, imagen[0].rect.height
        self.renderizadas += 1
        return nombre, max(round(ancho_pt), 1), max(round(alto_pt), 1)


def _sustituir_display(texto: str, renderizador: RenderizadorFormulas) -> str:
    """Reemplaza los bloques $$...$$ por imagenes centradas."""

    def reemplazo(coincidencia: re.Match) -> str:
        resultado = renderizador.renderizar(coincidencia.group(1), PUNTOS_DISPLAY)
        if resultado is None:
            cuerpo = html_lib.escape(coincidencia.group(1).strip())
            return f'\n\n<p class="formula"><code>{cuerpo}</code></p>\n\n'
        nombre, ancho, alto = resultado
        return (
            f'\n\n<p class="formula"><img src="{nombre}" '
            f'width="{ancho}" height="{alto}"></p>\n\n'
        )

    return re.sub(r"\$\$(.+?)\$\$", reemplazo, texto, flags=re.DOTALL)


def _sustituir_inline(texto: str, renderizador: RenderizadorFormulas) -> str:
    """Reemplaza los $...$ por imagenes alineadas con el texto."""

    def reemplazo(coincidencia: re.Match) -> str:
        resultado = renderizador.renderizar(coincidencia.group(1), PUNTOS_INLINE)
        if resultado is None:
            return f"<code>{html_lib.escape(coincidencia.group(1))}</code>"
        nombre, ancho, alto = resultado
        return f'<img src="{nombre}" width="{ancho}" height="{alto}">'

    return re.sub(r"(?<!\$)\$([^$\n]+?)\$(?!\$)", reemplazo, texto)


def _copiar_figuras() -> list[str]:
    """Copia las figuras de `artifacts/figures/` al directorio de armado."""
    origen = config.DIR_ARTIFACTS / "figures"
    if not origen.exists():
        raise FileNotFoundError(
            f"No existe {origen}. Corra `python main.py` antes de generar el PDF."
        )
    copiadas = []
    for figura in sorted(origen.glob("*.png")):
        shutil.copy2(figura, DIR_ARMADO / figura.name)
        copiadas.append(figura.name)
    return copiadas


def _ajustar_imagenes_de_figuras(texto_html: str) -> str:
    """Fija el ancho de las figuras del reporte para que quepan en la pagina."""
    return re.sub(
        r'<img alt="([^"]*)" src="(fig\d+[^"]*\.png)"\s*/?>',
        r'<p class="figura"><img src="\2" width="490"></p>',
        texto_html,
    )


def construir() -> dict:
    """Genera el PDF y devuelve un resumen de la construccion."""
    import markdown

    if DIR_ARMADO.exists():
        shutil.rmtree(DIR_ARMADO)
    DIR_ARMADO.mkdir(parents=True)

    figuras = _copiar_figuras()
    texto = RUTA_MARKDOWN.read_text(encoding="utf-8")

    renderizador = RenderizadorFormulas(DIR_ARMADO)
    texto = _sustituir_display(texto, renderizador)
    texto = _sustituir_inline(texto, renderizador)

    cuerpo = markdown.markdown(
        texto, extensions=["tables", "sane_lists", "attr_list"]
    )
    cuerpo = _ajustar_imagenes_de_figuras(cuerpo)
    documento_html = f"<html><head>{ESTILO}</head><body>{cuerpo}</body></html>"
    (DIR_ARMADO / "reporte.html").write_text(documento_html, encoding="utf-8")

    historia = pymupdf.Story(
        html=documento_html, archive=pymupdf.Archive(str(DIR_ARMADO))
    )
    rectangulo = pymupdf.paper_rect("letter")
    area = rectangulo + (54, 54, -54, -62)

    # DocumentWriter escribe los flujos de contenido SIN comprimir, lo que
    # inflaba el PDF a ~52 MB. Se escribe a un intermedio y se vuelve a guardar
    # con compresion y recoleccion de objetos huerfanos.
    intermedio = DIR_ARMADO / "reporte_sin_comprimir.pdf"
    escritor = pymupdf.DocumentWriter(str(intermedio))
    mas = 1
    paginas = 0
    while mas:
        dispositivo = escritor.begin_page(rectangulo)
        mas, _ = historia.place(area)
        historia.draw(dispositivo)
        escritor.end_page()
        paginas += 1
        if paginas > 200:
            raise RuntimeError("El reporte excede 200 paginas: revise el HTML.")
    escritor.close()

    bytes_sin_comprimir = intermedio.stat().st_size
    with pymupdf.open(intermedio) as documento:
        documento.set_metadata(
            {
                "title": "Lab 03 MyST - Arbitraje estadistico con cointegracion y filtros de Kalman",
                "author": "Rogelio Adrian Arroyo Valencia; Ernesto Andres Gonzalez Lomeli",
                "subject": "Microestructuras y Sistemas de Trading - ITESO - Equipo 1 - Nivel B",
                "creator": "docs/build_reporte.py",
            }
        )
        documento.save(
            str(RUTA_PDF), garbage=4, deflate=True, deflate_images=True,
            deflate_fonts=True, clean=True,
        )

    return {
        "bytes_sin_comprimir": bytes_sin_comprimir,
        "paginas": paginas,
        "bytes": RUTA_PDF.stat().st_size,
        "formulas_renderizadas": renderizador.renderizadas,
        "formulas_fallidas": renderizador.fallidas,
        "figuras_copiadas": figuras,
    }


def main() -> None:
    """Construye el PDF e imprime el resumen."""
    resumen = construir()
    print(f"PDF: {RUTA_PDF}")
    print(f"  paginas: {resumen['paginas']}")
    print(
        f"  tamano: {resumen['bytes'] / 1024:.0f} KB "
        f"(sin comprimir habria sido {resumen['bytes_sin_comprimir'] / 1024 / 1024:.0f} MB)"
    )
    print(f"  formulas renderizadas: {resumen['formulas_renderizadas']}")
    print(f"  figuras disponibles: {len(resumen['figuras_copiadas'])}")
    if resumen["formulas_fallidas"]:
        print(f"  FORMULAS FALLIDAS ({len(resumen['formulas_fallidas'])}):")
        for fallida in resumen["formulas_fallidas"]:
            print(f"    - {fallida}")
    else:
        print("  sin formulas fallidas")


if __name__ == "__main__":
    main()
