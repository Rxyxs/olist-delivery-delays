[[English](README.md) ] | [Español ]

# ¿Cuánto le cuesta a un marketplace una entrega atrasada?

![CI](https://github.com/Rxyxs/olist-delivery-delays/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)

96.470 pedidos reales entregados de Olist, un marketplace brasileño (2016-2018), unidos a
partir de nueve tablas: clientes, vendedores, ítems, productos, pagos, reseñas y geolocalización.
Tres preguntas, cada una respondida con el método que necesita:

1. **¿Cuánto baja la nota de la reseña una entrega atrasada?** Una pregunta causal: los pedidos
   atrasados son distintos de los puntuales antes de que algo salga mal.
2. **¿Se puede saber al momento de la compra qué pedidos se van a atrasar?** Una pregunta de
   predicción, validada en meses posteriores.
3. **¿Un primer pedido atrasado hace que el cliente no vuelva?** Una pregunta de supervivencia.

## Resultados en breve

| Pregunta | Respuesta |
|---|---|
| Efecto de llegar tarde sobre la reseña (1-5) | **-1,97 puntos** (IC 95%: -2,02 a -1,93), doblemente robusto |
| ... 1-3 días de atraso | -0,98 |
| ... 4-7 días de atraso | -2,15 |
| ... 8 días o más | cerca de -2,5, sin más daño |
| Atrasos detectados en el 10% de pedidos más riesgosos al comprar | 23% (al azar: 10%), AUC 0,69 |
| Efecto de un primer pedido atrasado sobre volver a comprar en 180 días | -0,4 pp (IC 95%: -1,0 a +0,2): no detectable |

## 1. Efecto de una entrega atrasada sobre la reseña

El 6,8% de los pedidos llega después de la fecha prometida. Su reseña promedio es 2,27 contra
4,29 de los puntuales, y el 54% recibe una estrella. Pero los pedidos atrasados viajan más lejos,
van a algunos estados más que a otros, llevan más ítems y se concentran en fechas peak, y todo
eso podría bajar la nota por sí solo. Cuatro estimadores, del ingenuo al doblemente robusto, para
el efecto sobre los pedidos atrasados (ATT):

| Estimador | Efecto sobre la reseña |
|---|---|
| Diferencia de medias | -2,02 |
| OLS con todas las covariables de compra | -1,98 |
| Ponderación por propensión inversa | -1,97 |
| **AIPW, LightGBM con cross-fitting de 5 folds** | **-1,97** (EE 0,022) |

Covariables: estado del cliente y del vendedor, distancia entre ambos, plazo prometido, precio,
flete, peso, número de ítems y de vendedores, categoría del producto, mes y día de la semana de
la compra, y el número de pedidos previos del vendedor.

**Ajustar casi no mueve la estimación.** Lo que hace distintos a los pedidos atrasados explica
casi nada de la caída de la nota; casi todo es el atraso mismo.

**Haría falta un confusor oculto muy fuerte para anularlo.** Siguiendo a Cinelli y Hazlett (2020),
una variable no observada tendría que explicar el 27% de la varianza residual tanto del atraso
como de la nota para llevar el efecto a cero. La covariable observada más fuerte, el plazo
prometido, explica el 1,1% del atraso y el 0,1% de la nota.

![Dosis-respuesta](docs/img/dose_response.png)

**El daño se concentra al principio.** Uno a tres días de atraso cuestan cerca de un punto; una
semana cuesta más de dos; después la nota ya no tiene mucho más que bajar.

![Solapamiento de propensión](docs/img/overlap.png)

Los pedidos atrasados tienen pedidos puntuales comparables en todo el rango de propensión, así
que la comparación no extrapola. Las propensiones se recortan en 1% y 99%.

### Una trampa que vale la pena nombrar: quedarse solo con reseñas posteriores a la entrega

El paso de limpieza obvio, descartar reseñas escritas antes de que llegara el paquete, aquí es un
error. Olist envía la encuesta cuando llega el producto *o* cuando vence la fecha prometida. En los
pedidos atrasados eso significa que la encuesta suele llegar antes que el paquete: **el 74% de los
pedidos atrasados se reseñó antes de la entrega**, por clientes que ya sabían que venía tarde.
Filtrarlos deja solo 1.621 de 6.337 pedidos atrasados y limita el atraso a 8 días, así que el
análisis mediría sobre todo atrasos leves.

## 2. Predecir atrasos al momento de la compra

Solo se usa información disponible al comprar. El historial del vendedor es la fracción de sus
pedidos anteriores que *ya habían sido entregados* tarde cuando se hizo este pedido, ajustada hacia
la tasa general. Entreno con pedidos hasta marzo de 2018 y evalúo abril a agosto de 2018:

| Modelo | AUC | PR-AUC | Atrasos en el 10% más riesgoso | en el 20% más riesgoso |
|---|---|---|---|---|
| Tasa de atraso del estado del cliente | 0,56 | 0,056 | 13% | 31% |
| Regresión logística | 0,67 | 0,082 | 21% | 37% |
| **LightGBM** | **0,69** | **0,086** | **23%** | **40%** |

![Captura de atrasos](docs/img/risk_capture.png)

La señal es real pero modesta: la mayoría de los atrasos vienen de eventos que no existen al
comprar. Además la tasa de atraso bajó de 7,9% en el período de entrenamiento a 4,4% en el de
prueba, por eso el corte es temporal. Las variables más útiles son el estado del cliente, el
historial del vendedor, la categoría y el plazo prometido. En la práctica el modelo sirve como
lista de prioridades (a quién avisar, qué envíos mejorar), no como garantía.

## 3. ¿Un primer pedido atrasado hace perder al cliente?

Solo el 3,0% de los clientes de Olist vuelve a comprar, así que hay pocos datos para esta
pregunta. Usando solo primeros pedidos:

- Modelo de Cox sobre el tiempo hasta el segundo pedido: hazard ratio 0,90 (IC 95%: 0,78-1,03,
  p = 0,12).
- AIPW sobre volver a comprar en 180 días (clientes con la ventana completa): 3,1% tras un primer
  pedido puntual, 2,6% tras uno atrasado; efecto ajustado -0,4 puntos (IC 95%: -1,0 a +0,2).

Las estimaciones puntuales apuntan a que el atraso hace perder clientes, pero los datos no
permiten distinguirlo de un efecto nulo. Un efecto real menor a cerca de un punto porcentual
(alrededor de 30% en términos relativos) pasaría inadvertido.

## Método

```
9 tablas de Olist ──► una fila por pedido entregado (data.py)
   │   confusores conocidos al comprar · tratamiento (atraso, días) · resultados
   │
   ├── causal.py     diferencia · OLS · IPW · AIPW (LightGBM con cross-fitting), dosis-respuesta
   │                 por banda de atraso, solapamiento, valor de robustez de Cinelli-Hazlett
   ├── risk.py       corte temporal, historial del vendedor sin mirar al futuro,
   │                 tasa por estado vs logística vs LightGBM
   └── retention.py  solo primeros pedidos: Cox sobre el tiempo al siguiente pedido, AIPW a 180 días
```

## Datos

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce),
Kaggle, licencia **CC BY-NC-SA 4.0**. Datos comerciales reales anonimizados. Descargados en octubre
de 2026; no se redistribuyen en este repositorio.

## Reproducir

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m olist.data        # descarga con ~/.kaggle/kaggle.json y arma data/orders.parquet
python -m olist.causal      # ~3 min
python -m olist.risk
python -m olist.retention
python -m olist.figures
pytest -q tests
```

## Limitaciones

- El "atraso" se mide contra la promesa de Olist. Una promesa conservadora escondería demoras que
  el cliente igual percibe.
- Algunos pedidos tienen más de una reseña; se usa la última respondida.
- La ausencia de confusores no observados es un supuesto. El análisis de sensibilidad lo acota,
  pero no descarta, por ejemplo, un problema de calidad del producto que además retrase el envío.

## Licencia

Código: MIT. Datos: CC BY-NC-SA 4.0, Olist (no se redistribuyen).
