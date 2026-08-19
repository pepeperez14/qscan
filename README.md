# qscan — escáner multiactivo en tres horizontes

Analiza la evolución de las cotizaciones del mayor número posible de activos
(acciones de EE.UU. y Europa, ETFs, cripto, materias primas, divisas, índices y
renta fija) y produce un ranking diario en tres horizontes: **corto** (1-3
semanas), **medio** (1-6 meses) y **largo** (6-24 meses).

Corre entero en GitHub Actions, gratis, y publica el informe en GitHub Pages.

---

## Lo primero: qué es y qué no es

El sistema usa **solo precio y volumen**. No lee resultados empresariales, ni
noticias, ni posicionamiento. Eso lo hace robusto (no hay datos que se revisen a
posteriori) y limitado (no sabe *por qué* algo se mueve).

El score **no es una predicción**. Es un ranking transversal: dice qué activos
están mejor posicionados hoy según un conjunto fijo de reglas, comparados con sus
iguales. Un percentil 99 significa "el mejor colocado del grupo según estas
reglas", no "va a subir".

**Antes de usar el ranking, mira la tabla de validación.** Está en el informe y
en `out/verdict.csv`. Si el t-stat de tu grupo y horizonte no supera 2, el orden
que produce el sistema no se distingue estadísticamente del azar, por muy
convincente que parezca la tabla del top-40. Escanear miles de activos con
decenas de indicadores siempre produce un top que parece brillante; ese es
precisamente el problema que la validación existe para detectar.

Esto no es asesoramiento de inversión.

---

## Puesta en marcha

```bash
git clone <tu-repo> && cd qscan
pip install -r requirements.txt
export PYTHONPATH=src

python -m qscan.cli universe            # construye el universo (~9.000 activos)
python -m qscan.cli update --years 8    # descarga OHLCV (la primera vez tarda)
python -m qscan.cli scan                # features + scores
python -m qscan.cli validate            # walk-forward
python -m qscan.cli report              # -> out/index.html
```

Para probarlo en 2 minutos sin descargar nada:

```bash
python tests/test_pipeline.py    # controles de calidad del motor
python tests/test_anomalies.py   # detector de datos rotos
python tests/test_calendar.py    # mezcla de calendarios (cripto 24/7 + bolsas)
python tests/demo_report.py      # informe de ejemplo con datos sintéticos
```

### En GitHub Actions

1. Sube el repo a GitHub.
2. *Settings → Pages → Source: GitHub Actions*.
3. *Actions → Escaneo diario → Run workflow* para la primera ejecución (la que
   construye el universo y descarga la historia completa; tarda bastante).

A partir de ahí corre solo. Dos disparos programados, de martes a sábado en UTC
(que es como se ven de lunes a viernes en hora local europea):

- **01:00 UTC — principal.** No se lanza justo tras el cierre de EE.UU. a
  propósito: Yahoo tarda un rato en consolidar los precios ajustados.
- **05:00 UTC — reintento.** No es un segundo escaneo. Si el principal terminó
  bien, se salta solo en segundos leyendo `.last_success`. Existe porque la
  descarga depende de Yahoo, que limita el ritmo y falla de vez en cuando; sin
  esa red, un fallo puntual te deja sin informe hasta el día siguiente.

Lanzarlo más veces al día no aportaría nada y haría daño: el sistema usa velas
diarias, y a media sesión la vela de hoy está incompleta. El cierre sería un
precio de mediodía, el volumen una fracción del real, y `vol_surge` saldría
artificialmente bajo en todo el universo.

El almacén de precios vive en la caché de Actions (límite 10 GB, de sobra), así
que las ejecuciones siguientes solo descargan lo nuevo.

**Recarga completa cada 30 días.** La actualización incremental sólo pide los
últimos días, pero Yahoo revisa la historia *hacia atrás*: cuando una empresa hace
un split o paga dividendo, reajusta la serie entera. Sin recarga, el almacén se
queda con el tramo antiguo en la escala vieja y el nuevo en la nueva, con un salto
falso justo en la frontera — el detector de anomalías lo caza y aparta el activo,
que es apagar un valor sano en lugar de arreglar el dato.

La recarga **descarta** las filas viejas de los símbolos que vuelve a bajar, en
vez de fusionarlas: fusionar deja el salto exactamente donde estaba.
`tests/test_recarga.py` reproduce un split 4:1 y comprueba las dos ramas —
fusionando, el activo queda marcado; sustituyendo, no.

Se dispara sola leyendo `data/.ultima_recarga`, sin tocar el workflow. Para
forzarla: `python -m qscan.cli update --full`. El periodo se cambia con
`--refresh-days`.

**Control de frescura.** Tras descargar, la ejecución falla en rojo si la última
cotización lleva más de 2 sesiones de retraso. No es una precaución teórica: en
agosto de 2026 el almacén se quedó anclado tres sesiones mientras las ejecuciones
seguían saliendo en verde, porque el modo de fallo real no fue un error sino
**silencio** — la descarga dejó de traer datos, la fusión no cambió nada y el
resto del sistema siguió calculando y publicando sobre precios viejos.

Una ejecución verde con datos rancios es peor que una roja, porque nadie la mira.
`tests/test_frescura.py` reproduce ese caso exacto y comprueba que ahora corta,
sin dar falsos positivos en festivos.

**Nada de fallos tragados.** Si la simulación de cartera se rompe, la ejecución
falla con la traza completa en vez de dejar un informe impecable con una cartera
que no avanza. Y `portfolio.commitear` dice explícitamente si consiguió guardar:
antes, un push fallido dejaba la simulación recalculándose cada día sin persistir
nunca, y el log no lo mencionaba.

Es el mismo patrón que este proyecto ya ha pagado tres veces: grupos que
desaparecían del ranking, precios que dejaban de actualizarse, y ahora esto. El
modo de fallo por defecto de un sistema automático no es el error, es el
**silencio**. Cada vez que algo pueda quedarse callado, hay que hacer que grite.

**Fuente de reserva.** Si la principal devuelve menos de la mitad de los símbolos
o no aporta fechas nuevas, se intenta Stooq antes de rendirse. Cubre acciones y
ETFs de EE.UU.; futuros, divisas e índices se dejan fuera a propósito, porque un
símbolo mal traducido no falla: devuelve la serie de otro activo.

**Coste**: en un repo público los minutos de Actions son ilimitados. En uno
privado el plan gratuito da 2.000 min/mes y una ejecución diaria completa se los
come; si lo quieres privado, reduce el universo con `--limit` o pasa a semanal.

---

## Arquitectura

```
universe.py   NASDAQ Trader + Wikipedia + ccxt  -> lista de activos
data.py       descarga por lotes -> almacén parquet incremental + control de calidad
indicators.py indicadores vectorizados sobre matrices fecha x símbolo
features.py   panel de ~35 características por activo y fecha
anomalies.py  detección de series rotas -> cuarentena antes de puntuar
scoring.py    z-scores winsorizados por grupo -> score por horizonte
validate.py   IC de Spearman, deciles, rotación, coste  -> veredicto
explain.py    capa de lectura: convierte el ranking en comentario
report.py     dashboard HTML autocontenido
```

Todo el cálculo es vectorizado sobre matrices anchas: 9.000 activos y 8 años de
historia se procesan en minutos, no en horas.

### Qué mide cada horizonte

| | Corto | Medio | Largo |
|---|---|---|---|
| Ventana | 1-3 semanas | 1-6 meses | 6-24 meses |
| Idea | ruptura con volumen, corregida por reversión de muy corto plazo | tendencia intermedia y fuerza relativa | momento largo y calidad de la tendencia |
| Pesos principales | Donchian 20, MACD, ADX, RSI(2) invertido | retorno 3m, fuerza relativa, cruce 50/200 | momento 12-1, ajuste de la tendencia, Sharpe |

Los pesos están en `scoring.py:WEIGHTS` y son el sitio natural donde intervenir.

---

## Decisiones de diseño que importan

**Comparación dentro del grupo.** Un Sharpe de 0,8 es excelente en materias
primas y mediocre en cripto. Puntuar todo contra la misma distribución produce un
ranking de clases de activo disfrazado de ranking de oportunidades.

**Winsorización antes de estandarizar.** Con miles de activos siempre hay colas
absurdas, y un solo z de 40 se come el peso de todo lo demás.

**Cobertura mínima de componentes.** Un activo con 3 de 9 features calculadas
puede encabezar el ranking por accidente aritmético. Se descarta.

**Filtro de liquidez.** Un valor que negocia 8.000 € al día no es una
oportunidad, es una trampa. Por defecto se exige 1 M$ de volumen mediano diario.

**Calidad de la tendencia, no solo pendiente.** `trendfit` es el R² de una
regresión log-lineal móvil: dos activos con la misma subida anual pero uno
ordenado y otro a trompicones no son la misma oportunidad, y casi ningún escáner
mira esto.

**Un solo calendario.** La cripto cotiza 365 días al año y las bolsas 252. Al
pivotarlas juntas, el índice común crece a ~726 filas por cada 2 años en vez de
520, y como todas las ventanas están expresadas en filas, una ventana de 252
dejaba de ser 12 meses para pasar a ser 8,3 meses **para todos los activos**.
Peor: esas 252 filas contenían sólo ~180 cotizaciones reales de una acción, así
que `slope_12m` y `trendfit_12m` salían NaN para el 100% de la renta variable y
las acciones desaparecían enteras del ranking de largo plazo, sin dar ningún
error. `data.to_business_calendar` alinea todo al calendario bursátil y
`tests/test_calendar.py` lo vigila.

**Errores estándar corregidos por solapamiento.** Con rebalanceo mensual y
horizonte de 12 meses, dos observaciones consecutivas comparten 11 meses de
retorno futuro. Usar el t-stat clásico lo infla por un factor cercano a √11: es
la razón número uno por la que un backtest parece significativo y luego no lo es.
Aquí se usa Newey-West.

---

## Las dos capas de IA, y la que falta a propósito

**Detección de anomalías** (`anomalies.py`). No predice nada: busca datos rotos.
Cotizaciones congeladas, splits sin ajustar, ticks malos que se revierten,
sesiones sin volumen, OHLC plano, feeds muertos y series duplicadas. Los activos
que superan severidad 4 quedan en cuarentena y **no entran en el ranking**,
porque un solo dato malo contamina los 35 indicadores de ese activo y lo cuela en
el top-40 con un +400% que nunca ocurrió. `tests/test_anomalies.py` inyecta cada
tipo de fallo y comprueba que lo encuentra: 7 de 7 con 0% de falsos positivos
sobre 184 series limpias.

Detalle de diseño que conviene conocer: sólo se buscan splits de 2:1 en adelante.
Un split 5:4 deja un salto del 20%, que en una cripto es un martes cualquiera; no
hay forma fiable de distinguirlo del movimiento real mirando sólo el precio, y se
prefiere no detectarlo antes que marcar como roto un dato bueno. Además, la
tolerancia de detección escala con la volatilidad del activo, porque el ratio
observado en un split no sale exacto: incorpora también el movimiento real de esa
sesión.

**Capa de lectura** (`explain.py`). Convierte el ranking en comentario legible.
La regla que la hace segura: **el modelo de lenguaje no calcula nada y no predice
nada**. La descomposición del score en aportaciones (`aportación = peso × z`) es
aritmética y sale de `scoring.contributions`; el modelo sólo verbaliza cifras que
ya existían. Si el veredicto de validación dice que no hay señal en ese grupo, el
comentario lo dice en la primera frase.

Se activa con el secreto `ANTHROPIC_API_KEY`. Sin él el informe sale igual, con
textos deterministas algo más secos y exactamente la misma información: es una
comodidad, no una dependencia.

**Lo que no está, y es deliberado:** un modelo de ML que sustituya el scoring
lineal. Tienes ~9.000 filas por fecha, pero están dominadas por un factor de
mercado común, así que la unidad de información independiente es la fecha, no el
activo-fecha: con 8 años de rebalanceo mensual son ~96 observaciones para ajustar
35 características. Un gradient boosting ahí no aprende el mercado, memoriza el
periodo. Y como las ventanas de retorno futuro se solapan, la validación cruzada
normal filtra información entre folds y da un resultado excelente y falso. Si
llega el momento, hace falta validación con purga y embargo, pesos por unicidad
de muestra, predecir el rango transversal en vez del retorno, y siempre contra el
baseline lineal en los mismos folds. Y antes de nada, saber cuánto IC saca el
baseline: "el modelo saca 0,05" no significa nada si el lineal sacaba 0,04 con
una décima parte de la complejidad.

## La descarga: el fallo que no da error

El modo de avería más caro que ha tenido este sistema no fue una excepción, fue
una merma silenciosa. Yahoo limita el ritmo de peticiones, y `yfinance` **no
lanza error cuando eso pasa**: escribe los fallos por consola y devuelve el lote
a medias. Con lotes de 200 símbolos y todos los hilos abiertos, los primeros
veintitantos lotes volvían enteros y a partir del cuarenta se perdían ~170 de
cada 200. Más de mil activos fuera del ranking, la ejecución en verde y un
informe con aspecto impecable.

Lo grave no es el número, es el sesgo: los símbolos que sobreviven no son una
muestra aleatoria del universo, son **los que se pidieron primero**. El ranking
se calculaba sobre un subconjunto ordenado por posición en la lista.

Cuatro defensas, en este orden:

1. **El control mira lo que volvió, no las excepciones.** Cada lote compara
   símbolos pedidos con símbolos devueltos. Es la única señal fiable cuando la
   fuente falla sin fallar.
2. **Ritmo adaptativo.** La pausa se multiplica por 1,8 cuando un lote vuelve
   incompleto (tope 30 s) y decae cuando vuelve entero. El ritmo se ajusta solo
   al límite real de ese día en vez de a una constante elegida a ojo.
3. **Pasada de rescate.** Lo que falte tras la primera vuelta se reintenta al
   final, después de dejar enfriar el contador dos minutos, en lotes pequeños y
   con pocos hilos. Con tope declarado: si faltan más de 2.500 símbolos se dice
   en el log cuántos se quedan sin reintentar, en vez de recortar en silencio.
4. **La reserva rellena huecos concretos.** Antes Stooq sólo entraba si Yahoo
   fallaba casi por completo — todo o nada — y por eso nunca entraba en el fallo
   real, que es parcial. Ahora se le piden exactamente los símbolos que faltan.

Y una precaución al empalmar fuentes: Yahoo ajusta por splits **y dividendos**,
Stooq sólo por splits. Concatenar los dos tramos sin más crea un escalón de
precio inventado —del orden del 25% en ocho años para un valor con dividendo
normal— que además dispara el detector de anomalías y acaba apartando un activo
sano. Por eso `empalmar_reserva` calcula el factor en la última fecha común y
reescala el tramo nuevo al nivel del almacén; si el factor sale fuera de 0,2-5
se deja el símbolo sin tocar, porque un empalme así significa que los dos
tickers no son el mismo activo.

Al final de cada actualización se registra la **cobertura de la última sesión**
—cuántos símbolos tienen la barra de hoy sobre el total del universo— y por
debajo del 75% se emite un ERROR. El número de filas del almacén no vale para
esto: sigue creciendo aunque medio universo se haya quedado sin cotización de
hoy. `tests/test_descarga.py` reproduce el escenario completo.

## La avería de la cartera: `float32` y un `except` demasiado amable

Vale la pena dejarlo escrito porque el patrón se repetirá. La simulación de
cartera estuvo semanas congelada en la misma fecha, con las mismas 75 órdenes
pendientes, mientras el informe se publicaba cada día impecable. La causa:

- el almacén guarda los precios en `float32` para que ocho años de historia
  quepan en la caché, así que todo lo que sale de las matrices de precios es
  `numpy.float32`, no `float`;
- `json.dumps` no sabe serializar `numpy.float32` y lanza `TypeError`;
- el fallo saltaba **al guardar**, es decir, después de simular el día entero
  correctamente: se calculaba todo y se perdía al escribirlo;
- y estaba dentro de un `except` que se limitaba a registrar un aviso.

Tres lecciones, aplicadas:

1. **Un `except` que traga una excepción tiene que fallar en rojo o gritar.** Se
   arregló antes de encontrar la causa, y fue lo que reveló la traza.
2. **Las pruebas tienen que usar los tipos de producción.** `test_cartera.py`
   construía sus precios sintéticos en `float64`, más cómodo, y por eso pasaba
   mientras el sistema real llevaba semanas roto. Ahora los convierte a
   `float32` a propósito: con el arreglo deshecho, la prueba falla.
3. **Serializar es un borde del sistema.** `portfolio.a_json` recorre la
   estructura entera y convierte tipos de numpy, fechas, NaN e infinitos, en vez
   de confiar en que lo que se guarda ya sea JSON válido.

## El motor está probado, el modelo no

`tests/test_pipeline.py` verifica tres cosas sobre datos sintéticos:

1. **Control negativo**: sobre paseos aleatorios sin deriva, el sistema informa
   de que no hay señal. Si aquí apareciera un t-stat alto, habría una fuga de
   información en el pipeline.
2. **Control positivo**: sobre series con momento inyectado, lo detecta. Si aquí
   no apareciera nada, el cálculo estaría roto.
3. **Ausencia de lookahead**: ninguna característica cambia de valor al añadirle
   datos posteriores.

Que estas tres pasen significa que **el motor calcula bien**. No significa que
las reglas de `WEIGHTS` funcionen en el mercado real: eso solo lo dice
`validate` sobre datos reales, y es una pregunta abierta hasta que la respondas
con varios años de historia.

## Simulación de cartera: cuatro escenarios

Cada día se simulan **cuatro carteras independientes, con 40.000 € cada una**,
para que sean comparables entre sí y contra el índice:

| Escenario | Posiciones | Rebalanceo | Idea |
|---|---|---|---|
| Corto | 15 | semanal | donde la validación encuentra mejor t-stat, al precio de rotar mucho |
| Medio | 20 | mensual | mejor equilibrio entre señal y coste |
| Largo | 20 | trimestral | costes mínimos; hoy sin evidencia de señal |
| Combinada | 20 | mensual | reparto entre los tres según la evidencia medida |

La **combinada** pondera cada horizonte por su t-stat por encima de 1: por debajo
de ahí no hay nada que distinga la señal del azar y no merece capital. Si ningún
horizonte llega, reparte a partes iguales y lo dice. Contrapartida honesta: los
pesos salen de la misma validación que mide el sistema, así que hay algo de
ajuste a los propios datos; se mitiga con un umbral duro en vez de optimizar
libremente, pero no desaparece.

### Optimización de costes: sólo se opera si compensa

Cada movimiento pasa por tres filtros antes de ejecutarse:

1. **Tamaño mínimo.** Con 1 € de comisión, una compra de 200 € paga un 0,5% antes
   de empezar. Se exige que la comisión no supere el 0,1% del importe, lo que fija
   un mínimo de 1.000 € por operación en Trade Republic.
2. **Banda de no operar.** Si el peso actual se desvía menos de un 25% del
   objetivo, se deja quieto. Reequilibrar por reequilibrar es pagar por nada.
3. **El beneficio esperado tiene que cubrir el coste con margen 1,5x.** Cambiar A
   por B sólo compensa si la diferencia de alfa supera el coste de vender A y
   comprar B. El margen existe porque el alfa es una estimación ruidosa: operar
   al filo es perder de media.

El alfa esperado sale de la aproximación de Grinold, **E[r] = IC × z × σ**, con el
IC que mide la propia validación para ese grupo y horizonte. Esto es lo que hace
que el sistema se adapte solo: si un grupo no tiene señal, su IC ronda cero, el
alfa ronda cero, ninguna rotación compensa la comisión y la cartera se queda
quieta. Dejar de operar cuando no sabes nada es una decisión, no una avería.

Un IC negativo se trata como cero. Invertir la señal porque el backtest salió al
revés es una de las formas más fáciles de sobreajustar.

El efecto medido sobre datos sintéticos, mismo periodo y mismas señales:

| | Sin optimizar | Con optimizador |
|---|---|---|
| Costes (corto, TR) | 505 € | **40 €** |
| Rentabilidad (corto, TR) | +4,61% | **+6,48%** |
| Rentabilidad (medio, TR) | +3,10% | **+7,56%** |

Y el control positivo: declarando un IC de 0,25, el mismo sistema pasa de 150 a
482 operaciones y sus costes vuelven a 505 €. Rota cuando hay algo que ganar y se
queda quieto cuando no. El informe enseña además la tabla de **operaciones
descartadas** con su motivo y el coste evitado: las operaciones que no se hacen no
dejan rastro en ningún sitio, así que sin esa tabla sería imposible saber si el
filtro trabaja o está apagado.

Cinco decisiones más la separan de un folleto:

- **Ejecución a la apertura del día siguiente.** El ranking sale del cierre de
  hoy y nadie puede comprar a ese precio. Ejecutar al precio con el que decides
  es la forma más silenciosa de inflar un backtest.
- **Comisión y horquilla.** La comisión se recuerda; la horquilla es la que se
  come el resultado al rotar. El efecto se ve solo: en la prueba, el escenario
  corto acumula 785 € de costes y el largo 79 € — diez veces más por rotar
  semanalmente en vez de trimestralmente.
- **Divisa.** Capital en euros, activos en dólares. En un año el euro-dólar se
  mueve más que muchas de las señales que perseguimos.
- **Índice de referencia.** Sin comparar contra comprar y esperar, un mercado
  alcista hace que cualquier estrategia parezca buena. Lo que importa es la
  columna *vs índice*, no la rentabilidad.
- **El pasado no se reescribe.** Cada día añade filas a `portfolio/curva.csv` y
  `portfolio/operaciones.csv`; nunca recalcula.

El informe lleva un **selector de periodo** (1 semana, 1 mes, 3, 6 meses, 1 año,
todo) que recalcula la rentabilidad acumulada de cada escenario en esa ventana y
redibuja los gráficos. La base es el valor al inicio del periodo, no el capital
inicial: si no, "3 meses" seguiría enseñando la rentabilidad desde el origen.
Los datos van embebidos y el cálculo lo hace el navegador, que es la única forma
de tener un selector real en una página estática.

También muestra siempre la **composición de cada cartera**: cada posición con su
valor, su peso y su plusvalía. Se enseña una sola composición porque ambos
brókers compran exactamente los mismos activos —el objetivo lo fija la señal, no
el bróker—; lo que cambia entre ellos es el coste.

El informe incluye además el **parte de operaciones**: qué habría que comprar y
vender en la próxima apertura, y qué se cruzó hoy con su comisión. En los días
sin rebalanceo aparece vacío, que es lo correcto: no operar es la postura por
defecto.

## Sesgo de supervivencia: lo único que hay que empezar hoy

El universo se construye a partir de los valores que cotizan **hoy**. Cualquier
validación sobre 8 años de historia incluye, por tanto, sólo a los que
sobrevivieron: las quiebras, exclusiones y fusiones fallidas no están. Eso hace
que el IC medido sea optimista, y como el IC es justo el número con el que
decides si el sistema sirve, el sesgo ataca al termómetro y no al enfermo.

No se puede arreglar hacia atrás; sólo se puede empezar a registrar. Por eso
`qscan universe` guarda una instantánea fechada en `data/snapshots/`. Dentro de
un año tendrás 52 fotos del universo y podrás reconstruir listas *point-in-time*
y medir cuánto se infla el IC. Cuesta un fichero comprimido por semana y es
irrecuperable si no se empieza ya.

## Extensiones naturales

- **Régimen de mercado**: el mismo indicador tiene IC muy distinto en tendencia
  que en un giro. Condicionar los pesos al régimen (volatilidad, dispersión
  transversal, amplitud) suele aportar más que cambiar de modelo, y con muchos
  menos parámetros.
- **Costes por clase de activo**: ahora hay 10 pb planos. La horquilla real va de
  1-2 pb en una mega cap a 50-100 pb en una small cap o una alt. Como el spread
  por deciles se concentra justo en lo ilíquido, el coste plano es optimista
  donde más duele.
- **Estabilidad temporal del IC**: un IC medio de 0,04 producido íntegramente por
  2020 no es una señal, es un evento. Conviene desglosarlo por año.
- **Registro de investigación**: cada vez que toques `WEIGHTS` estás haciendo un
  test más. Anotar cuántas configuraciones has probado es lo que permite deflactar
  el resultado y no engañarte a lo largo de los meses.
- **Datos intradía** para el horizonte corto: el diario se queda justo.
