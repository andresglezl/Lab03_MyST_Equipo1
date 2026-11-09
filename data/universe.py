"""Universo de candidatos pre-registrado y su bitacora de registro (seccion 3.2.1).

Este archivo es la defensa del equipo frente al problema de pruebas multiples.
Cada pool se congela ANTES de calcular sus estadisticos, y el campo
`momento_registro` deja asentado en que momento del proyecto entro al universo
y por que. Esa bitacora es parte del entregable: lo que invalida un hallazgo no
es agregar un pool, es agregarlo y no decirlo.

TODOS los pools listados en `POOLS_EVALUADOS` se evaluan en cada corrida, de
modo que la tabla de grupos probados y el total `m` son deterministas y
reproducibles con un solo comando.

Bitacora de registro
--------------------
1. **A1 (E&P / integradas)** y **A2 (refinacion)**: pools originales, registrados
   antes de calcular cualquier estadistico. Universo "cadena de valor del crudo".
2. **A3 (servicios petroleros)**: registrado como RESERVA desde el inicio, con la
   regla declarada de activarse solo si A1 o A2 no producian ningun grupo que
   pasara Johansen. Ningun grupo de A1 paso, asi que A3 se activo y se evaluo.
   Tampoco paso.
3. **B1 (bancos canadienses)**: registrado DESPUES de que A1 y A3 fallaran, con
   el unico fin de poder construir el portafolio de Risk Parity de nivel B, que
   exige al menos dos grupos cointegrados disjuntos. Se evaluo con exactamente
   los mismos criterios y umbrales que los pools anteriores, sin relajar
   ninguno. Esto es una ampliacion del universo documentada, no una busqueda de
   especificacion: no se reajusto ningun criterio para que A1 o A3 pasaran.
   Consecuencia honesta: `m` sube de 21 a 26 y con el sube el numero esperado de
   falsos positivos, y asi se reporta.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Tamano de grupo exigido = 2 * n integrantes = 2 * 2 = 4 (seccion 3.1).
ACTIVOS_POR_GRUPO = 4


@dataclass(frozen=True)
class PoolCandidatos:
    """Un pool pre-registrado del que se extraen los grupos candidatos.

    Attributes:
        identificador: clave corta usada en nombres de archivo de `artifacts/`.
        nombre: etiqueta legible para figuras y reporte.
        tickers: pool completo; los grupos candidatos son sus subconjuntos de
            tamano `ACTIVOS_POR_GRUPO`.
        tendencia_comun: la variable economica que induce la tendencia
            estocastica compartida.
        justificacion: por que esperamos cointegracion ANTES de probarla.
        momento_registro: cuando entro este pool al universo y por que.
    """

    identificador: str
    nombre: str
    tickers: tuple[str, ...]
    tendencia_comun: str
    justificacion: str
    momento_registro: str


POOL_A1 = PoolCandidatos(
    identificador="A1",
    nombre="E&P / integradas",
    tickers=("XOM", "CVX", "COP", "OXY", "EOG", "DVN"),
    tendencia_comun="Nivel del crudo (WTI / Brent)",
    justificacion=(
        "Las seis monetizan reservas probadas de hidrocarburos. El valor "
        "presente de esas reservas es una funcion monotona de la misma curva "
        "de precios del crudo, de modo que sus capitalizaciones comparten un "
        "unico factor estocastico no estacionario. Las diferencias entre ellas "
        "(costo de extraccion, apalancamiento, mezcla de cuencas) son "
        "estacionarias y por tanto candidatas a ser el spread reversivo."
    ),
    momento_registro="Original. Registrado antes de calcular cualquier estadistico.",
)

POOL_A2 = PoolCandidatos(
    identificador="A2",
    nombre="Refinacion",
    tickers=("VLO", "MPC", "PSX", "PBF", "DINO"),
    tendencia_comun="Margen de refinacion (crack spread 3-2-1)",
    justificacion=(
        "Las cinco compran crudo y venden gasolina y destilado en regiones "
        "PADD solapadas, con capacidad de procesamiento comparable y acceso a "
        "los mismos ductos. Su utilidad depende de un diferencial comun, no "
        "del nivel del crudo, por lo que comparten una tendencia estocastica "
        "distinta a la de A1. La capacidad de refinacion en EE.UU. es casi "
        "fija (no se construyen refinerias nuevas), lo que hace que el reparto "
        "de margen entre ellas sea mas estable que en upstream."
    ),
    momento_registro="Original. Registrado antes de calcular cualquier estadistico.",
)

POOL_A3 = PoolCandidatos(
    identificador="A3",
    nombre="Servicios petroleros",
    tickers=("SLB", "HAL", "NOV", "RIG"),
    tendencia_comun="Ciclo global de capex upstream",
    justificacion=(
        "Las cuatro venden equipo y servicios al mismo cliente agregado (los "
        "operadores de A1) y su ingreso se mueve con el presupuesto de capital "
        "de perforacion, no con el precio del crudo directamente. Al tener "
        "solo cuatro nombres, el pool aporta un unico grupo candidato y por "
        "tanto ninguna capacidad de seleccion interna."
    ),
    momento_registro=(
        "Reserva declarada desde el inicio, con la regla de activarse solo si "
        "A1 o A2 no producian grupos que pasaran Johansen. Se activo porque "
        "ningun grupo de A1 paso."
    ),
)

POOL_B1 = PoolCandidatos(
    identificador="B1",
    nombre="Bancos canadienses listados en NYSE",
    tickers=("RY", "TD", "BNS", "BMO", "CM"),
    tendencia_comun=(
        "Ciclo de tasas del Banco de Canada y mercado hipotecario domestico"
    ),
    justificacion=(
        "Son cinco bancos DISTINTOS, no el mismo instrumento en dos bolsas: "
        "cada uno esta cross-listed individualmente en NYSE y TSX, pero entre "
        "ellos no comparten flujos de efectivo ni derechos sobre los mismos "
        "activos. Lo que comparten es el entorno: el mismo regulador "
        "prudencial (OSFI) con las mismas reglas de capital y de provisiones, "
        "el mismo mercado hipotecario domestico concentrado y con seguro "
        "publico (CMHC), y el mismo ciclo de politica monetaria del Banco de "
        "Canada que fija su margen financiero. Ese entorno comun es el "
        "mecanismo que induce la tendencia estocastica compartida, y es "
        "nombrable y falsable: si cambiara el regimen regulatorio o el ciclo "
        "de tasas se divorciara entre ellos, la relacion deberia romperse."
    ),
    momento_registro=(
        "Agregado DESPUES de que A1 y A3 fallaran, para poder construir el "
        "portafolio de Risk Parity de nivel B, que exige al menos dos grupos "
        "cointegrados disjuntos. Evaluado con exactamente los mismos criterios "
        "y umbrales, sin relajar ninguno."
    ),
)

#: Todos los pools que el embudo evalua en cada corrida.
POOLS_EVALUADOS: tuple[PoolCandidatos, ...] = (POOL_A1, POOL_A2, POOL_A3, POOL_B1)


def tickers_del_universo() -> list[str]:
    """Devuelve la lista ordenada y sin repeticiones de tickers a descargar.

    Returns:
        Tickers de todos los pools evaluados, ordenados alfabeticamente.
    """
    return sorted({ticker for pool in POOLS_EVALUADOS for ticker in pool.tickers})


def pool_por_identificador(identificador: str) -> PoolCandidatos:
    """Devuelve el pool con el identificador dado.

    Args:
        identificador: clave corta del pool, p. ej. 'A2'.

    Returns:
        El pool correspondiente.

    Raises:
        KeyError: si no existe un pool con ese identificador.
    """
    for pool in POOLS_EVALUADOS:
        if pool.identificador == identificador:
            return pool
    raise KeyError(f"No existe el pool {identificador!r}.")


def verificar_pools_disjuntos(
    pools: tuple[PoolCandidatos, ...] = POOLS_EVALUADOS,
) -> None:
    """Valida que ningun ticker aparezca en dos pools.

    La seccion 3.7 exige que los grupos operados sean disjuntos en activos. Si
    los pools ya son disjuntos, cualquier par de grupos tomados de pools
    distintos lo es automaticamente.

    Raises:
        ValueError: si algun ticker se repite entre pools.
    """
    vistos: dict[str, str] = {}
    for pool in pools:
        for ticker in pool.tickers:
            if ticker in vistos:
                raise ValueError(
                    f"{ticker} aparece en los pools {vistos[ticker]} y "
                    f"{pool.identificador}: los pools deben ser disjuntos."
                )
            vistos[ticker] = pool.identificador


def total_grupos_candidatos() -> int:
    """Numero total de grupos que el embudo evalua, es decir `m`.

    Returns:
        Suma de C(|pool|, 4) sobre todos los pools evaluados.
    """
    from math import comb

    return sum(comb(len(pool.tickers), ACTIVOS_POR_GRUPO) for pool in POOLS_EVALUADOS)
