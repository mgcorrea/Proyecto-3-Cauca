from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Cargar los datos
file_path = '../datos_limpios.csv'  # Cambia esto si tu archivo está en otra ubicación
datos = pd.read_csv(file_path)

def predecir_puntaje(input_data):
    """
    Función para predecir el puntaje (PUNT_GLOBAL) a partir de los datos de entrada,
    utilizando el modelo entrenado con codificación one-hot.
    """
    # Cargar el escalador y el modelo
    scaler = joblib.load('../Tablero/scaler.pkl')  # Asegúrate de que 'scaler.pkl' esté en la misma ruta
    modelf = load_model('../Tablero/modelf.keras')  # Asegúrate de que 'modelo.keras' esté en la misma ruta
    
    # Columnas esperadas por el modelo
    expected_columns = ['COLE_AREA_UBICACION_URBANO', 'COLE_BILINGUE_S',
       'COLE_CALENDARIO_B', 'COLE_CALENDARIO_OTRO', 'COLE_CARACTER_NO APLICA',
       'COLE_CARACTER_TÉCNICO', 'COLE_CARACTER_TÉCNICO/ACADÉMICO',
       'COLE_GENERO_MASCULINO', 'COLE_GENERO_MIXTO', 'COLE_JORNADA_MAÑANA',
       'COLE_JORNADA_NOCHE', 'COLE_JORNADA_SABATINA', 'COLE_JORNADA_TARDE',
       'COLE_MCPIO_UBICACION_ARGELIA', 'COLE_MCPIO_UBICACION_BALBOA',
       'COLE_MCPIO_UBICACION_BOLIVAR', 'COLE_MCPIO_UBICACION_BUENOS AIRES',
       'COLE_MCPIO_UBICACION_CAJIBIO', 'COLE_MCPIO_UBICACION_CALDONO',
       'COLE_MCPIO_UBICACION_CALOTO', 'COLE_MCPIO_UBICACION_CORINTO',
       'COLE_MCPIO_UBICACION_EL TAMBO', 'COLE_MCPIO_UBICACION_FLORENCIA',
       'COLE_MCPIO_UBICACION_GUACHENÉ', 'COLE_MCPIO_UBICACION_GUAPI',
       'COLE_MCPIO_UBICACION_INZA', 'COLE_MCPIO_UBICACION_JAMBALO',
       'COLE_MCPIO_UBICACION_LA SIERRA', 'COLE_MCPIO_UBICACION_LA VEGA',
       'COLE_MCPIO_UBICACION_LOPEZ (MICAY)', 'COLE_MCPIO_UBICACION_MERCADERES',
       'COLE_MCPIO_UBICACION_MIRANDA', 'COLE_MCPIO_UBICACION_MORALES',
       'COLE_MCPIO_UBICACION_PADILLA',
       'COLE_MCPIO_UBICACION_PAEZ (BELALCAZAR)',
       'COLE_MCPIO_UBICACION_PATIA(EL BORDO)', 'COLE_MCPIO_UBICACION_PIENDAMO',
       'COLE_MCPIO_UBICACION_POPAYAN', 'COLE_MCPIO_UBICACION_PUERTO TEJADA',
       'COLE_MCPIO_UBICACION_PURACE (COCONUCO)', 'COLE_MCPIO_UBICACION_ROSAS',
       'COLE_MCPIO_UBICACION_SAN SEBASTIAN', 'COLE_MCPIO_UBICACION_SANTA ROSA',
       'COLE_MCPIO_UBICACION_SANTANDER DE QUILICHAO',
       'COLE_MCPIO_UBICACION_SILVIA',
       'COLE_MCPIO_UBICACION_SOTARA (PAISPAMBA)',
       'COLE_MCPIO_UBICACION_SUAREZ', 'COLE_MCPIO_UBICACION_SUCRE',
       'COLE_MCPIO_UBICACION_TIMBIO', 'COLE_MCPIO_UBICACION_TIMBIQUI',
       'COLE_MCPIO_UBICACION_TORIBIO', 'COLE_MCPIO_UBICACION_TOTORO',
       'COLE_MCPIO_UBICACION_VILLA RICA', 'COLE_NATURALEZA_OFICIAL',
       'ESTU_DEPTO_RESIDE_CAUCA', 'ESTU_DEPTO_RESIDE_CHOCO',
       'ESTU_DEPTO_RESIDE_HUILA', 'ESTU_DEPTO_RESIDE_NARIÑO',
       'ESTU_DEPTO_RESIDE_SAN ANDRES', 'ESTU_DEPTO_RESIDE_VALLE',
       'ESTU_GENERO_M', 'FAMI_EDUCACIONMADRE_Educación profesional incompleta',
       'FAMI_EDUCACIONMADRE_Ninguno', 'FAMI_EDUCACIONMADRE_No sabe',
       'FAMI_EDUCACIONMADRE_Postgrado',
       'FAMI_EDUCACIONMADRE_Primaria completa',
       'FAMI_EDUCACIONMADRE_Primaria incompleta',
       'FAMI_EDUCACIONMADRE_Secundaria (Bachillerato) completa',
       'FAMI_EDUCACIONMADRE_Secundaria (Bachillerato) incompleta',
       'FAMI_EDUCACIONMADRE_Técnica o tecnológica completa',
       'FAMI_EDUCACIONMADRE_Técnica o tecnológica incompleta',
       'FAMI_EDUCACIONPADRE_Educación profesional incompleta',
       'FAMI_EDUCACIONPADRE_Ninguno', 'FAMI_EDUCACIONPADRE_No sabe',
       'FAMI_EDUCACIONPADRE_Postgrado',
       'FAMI_EDUCACIONPADRE_Primaria completa',
       'FAMI_EDUCACIONPADRE_Primaria incompleta',
       'FAMI_EDUCACIONPADRE_Secundaria (Bachillerato) completa',
       'FAMI_EDUCACIONPADRE_Secundaria (Bachillerato) incompleta',
       'FAMI_EDUCACIONPADRE_Técnica o tecnológica completa',
       'FAMI_EDUCACIONPADRE_Técnica o tecnológica incompleta',
       'FAMI_ESTRATOVIVIENDA_Estrato 2', 'FAMI_ESTRATOVIVIENDA_Estrato 3',
       'FAMI_ESTRATOVIVIENDA_Estrato 4', 'FAMI_ESTRATOVIVIENDA_Estrato 5',
       'FAMI_ESTRATOVIVIENDA_Estrato 6', 'FAMI_TIENECOMPUTADOR_Si',
       'FAMI_TIENEINTERNET_Si']

    # Inicializar el vector con ceros
    input_vector = {col: 0 for col in expected_columns}

    # Mapear los datos de entrada a las columnas dummies
    for key, value in input_data.items():
        column_name = f"{key}_{value}"  # Nombre de la columna dummy
        if column_name in input_vector:
            input_vector[column_name] = 1

    # Convertir a numpy array
    input_array = np.array([list(input_vector.values())])

    # Escalar los datos de entrada
    input_scaled = scaler.transform(input_array)

    # Realizar la predicción
    puntaje_predicho = modelf.predict(input_scaled)

    return puntaje_predicho[0][0]


# Competencias disponibles
competencias = [
    'PUNT_INGLES', 'PUNT_MATEMATICAS', 'PUNT_SOCIALES_CIUDADANAS',
    'PUNT_C_NATURALES', 'PUNT_LECTURA_CRITICA'
]

# Columnas categóricas para Análisis Socioeconómico
columnas_socioeconomicas = [
    'ESTU_MCPIO_RESIDE', 'ESTU_NACIONALIDAD', 'ESTU_PAIS_RESIDE',
    'FAMI_EDUCACIONMADRE', 'FAMI_EDUCACIONPADRE', 'FAMI_ESTRATOVIVIENDA',
    'FAMI_TIENEAUTOMOVIL', 'FAMI_TIENECOMPUTADOR', 'FAMI_TIENEINTERNET', 'FAMI_TIENELAVADORA'
]

# Columnas categóricas para Colegios
columnas_colegios = [
    'COLE_AREA_UBICACION', 'COLE_BILINGUE', 'COLE_CALENDARIO',
    'COLE_CARACTER', 'COLE_DEPTO_PRESENTACION', 'COLE_GENERO',
    'COLE_JORNADA', 'COLE_MCPIO_UBICACION'
]

# Crear la aplicación Dash
app = Dash(__name__)
app.title = "Tablero Saber 11"

# Layout
app.layout = html.Div(
    children=[
        html.H1('Tablero de Análisis de Pruebas Saber 11', style={'textAlign': 'center'}),
        dcc.Tabs(
            className='custom-tabs',
            children=[
                # Tab: Análisis Colegios
                dcc.Tab(label='Análisis Colegios', className='custom-tab', selected_className='custom-tab--selected', children=[
                    html.Div([
                        html.H3('Seleccione una categoría:'),
                        dcc.Dropdown(
                            id='dropdown_colegios',
                            options=[{'label': col, 'value': col} for col in columnas_colegios],
                            value='COLE_AREA_UBICACION',
                            className='custom-dropdown'
                        ),
                        dcc.Graph(id='graph_colegios_bar'),  # Gráfico de barras
                        dcc.Graph(id='graph_colegios_box')   # Gráfico de caja
                    ])
                ]),

                # Tab: Análisis de Competencias
                dcc.Tab(label='Análisis de Competencias', className='custom-tab', selected_className='custom-tab--selected', children=[
                    html.Div([
                        html.H3('Seleccione una competencia:'),
                        dcc.Dropdown(
                            id='dropdown_competencias',
                            options=[{'label': competencia, 'value': competencia} for competencia in competencias],
                            value='PUNT_INGLES',
                            className='custom-dropdown'
                        ),
                        dcc.Graph(id='graph_competencias_1'),  # Relación puntaje global vs competencia
                        dcc.Graph(id='graph_competencias_2'),  # Estadísticas descriptivas
                        dcc.Graph(id='graph_competencias_3')   # Densidad y distribución
                    ])
                ]),

                # Tab: Análisis Socioeconómico
                dcc.Tab(label='Análisis Socioeconómico', className='custom-tab', selected_className='custom-tab--selected', children=[
                    html.Div([
                        html.H3('Seleccione una categoría:'),
                        dcc.Dropdown(
                            id='dropdown_socioeconomico',
                            options=[{'label': col, 'value': col} for col in columnas_socioeconomicas],
                            value='FAMI_ESTRATOVIVIENDA',
                            className='custom-dropdown'
                        ),
                        dcc.Graph(id='graph_socioeconomico_bar'),  # Gráfico de barras
                        dcc.Graph(id='graph_socioeconomico_box')   # Gráfico de caja
                    ])
                ]),
                  # Tab: Predicciones
                dcc.Tab(label='Predicciones', className='custom-tab', selected_className='custom-tab--selected', children=[
                    html.Div([
                        html.H3('Predicción del Puntaje Global'),
                        html.Div([
                            html.Label('COLE_AREA_UBICACION:'),
                                dcc.Dropdown(
                                    id='input_cole_area',
                                    options=[
                                        {'label': 'Rural', 'value': 'RURAL'},
                                        {'label': 'Urbano', 'value': 'URBANO'}
                                    ],
                                    placeholder='Seleccione una opción',
                                    className='custom-dropdown'
                                ),

                            html.Label('COLE_BILINGUE:'),
                                dcc.Dropdown(
                                    id='input_cole_bilingue',
                                    options=[
                                        {'label': 'Sí', 'value': 'S'},
                                        {'label': 'No', 'value': 'N'}
                                    ],
                                    placeholder='Seleccione una opción',
                                    className='custom-dropdown'
                                ),

                           html.Label('COLE_CALENDARIO:'),
                                dcc.Dropdown(
                                    id='input_cole_calendario',
                                    options=[
                                        {'label': 'A', 'value': 'A'},
                                        {'label': 'B', 'value': 'B'},
                                        {'label': 'OTRO', 'value': 'OTRO'}
                                    ],
                                    placeholder='Seleccione una opción',
                                    className='custom-dropdown'
                                ),

                           html.Label('COLE_CARACTER:'),
                            dcc.Dropdown(
                                id='input_cole_caracter',
                                options=[
                                    {'label': 'TÉCNICO', 'value': 'TÉCNICO'},
                                    {'label': 'ACADÉMICO', 'value': 'ACADÉMICO'},
                                    {'label': 'NO APLICA', 'value': 'NO APLICA'},
                                    {'label': 'TÉCNICO/ACADÉMICO', 'value': 'TÉCNICO/ACADÉMICO'}
                                ],
                                placeholder='Seleccione una opción',
                                className='custom-dropdown'
                            ),
                            html.Label('COLE_GENERO:'),
                                dcc.Dropdown(
                                    id='input_cole_genero',
                                    options=[
                                        {'label': 'FEMENINO', 'value': 'FEMENINO'},
                                        {'label': 'MIXTO', 'value': 'MIXTO'},
                                        {'label': 'MASCULINO', 'value': 'MASCULINO'}
                                    ],
                                    placeholder='Seleccione una opción',
                                    className='custom-dropdown'
                                ),
                            html.Label('COLE_JORNADA:'),
                            dcc.Dropdown(
                                id='input_cole_jornada',
                                options=[
                                    {'label': 'MAÑANA', 'value': 'MAÑANA'},
                                    {'label': 'COMPLETA', 'value': 'COMPLETA'},
                                    {'label': 'TARDE', 'value': 'TARDE'},
                                    {'label': 'SABATINA', 'value': 'SABATINA'},
                                    {'label': 'NOCHE', 'value': 'NOCHE'},
                                    {'label': 'ÚNICA', 'value': 'ÚNICA'}
                                ],
                                placeholder='Seleccione una opción',
                                className='custom-dropdown'
                            ),
                            html.Label('COLE_MCPIO_UBICACION:'),
                            dcc.Dropdown(
                                id='input_cole_mcpio',
                                options=[
                                    {'label': 'POPAYÁN', 'value': 'POPAYÁN'},
                                    {'label': 'TORIBÍO', 'value': 'TORIBÍO'},
                                    {'label': 'POPAYAN', 'value': 'POPAYAN'},
                                    {'label': 'EL TAMBO', 'value': 'EL TAMBO'},
                                    {'label': 'PAEZ (BELALCAZAR)', 'value': 'PAEZ (BELALCAZAR)'},
                                    {'label': 'SANTANDER DE QUILICHAO', 'value': 'SANTANDER DE QUILICHAO'},
                                    {'label': 'BUENOS AIRES', 'value': 'BUENOS AIRES'},
                                    {'label': 'SUCRE', 'value': 'SUCRE'},
                                    {'label': 'ARGELIA', 'value': 'ARGELIA'},
                                    {'label': 'ALMAGUER', 'value': 'ALMAGUER'},
                                    {'label': 'PURACÉ', 'value': 'PURACÉ'},
                                    {'label': 'MIRANDA', 'value': 'MIRANDA'},
                                    {'label': 'JAMBALÓ', 'value': 'JAMBALÓ'},
                                    {'label': 'INZÁ', 'value': 'INZÁ'},
                                    {'label': 'LA SIERRA', 'value': 'LA SIERRA'},
                                    {'label': 'CORINTO', 'value': 'CORINTO'},
                                    {'label': 'GUACHENÉ', 'value': 'GUACHENÉ'},
                                    {'label': 'TORIBIO', 'value': 'TORIBIO'},
                                    {'label': 'MERCADERES', 'value': 'MERCADERES'},
                                    {'label': 'LOPEZ (MICAY)', 'value': 'LOPEZ (MICAY)'},
                                    {'label': 'MORALES', 'value': 'MORALES'},
                                    {'label': 'VILLA RICA', 'value': 'VILLA RICA'},
                                    {'label': 'LA VEGA', 'value': 'LA VEGA'},
                                    {'label': 'PURACE (COCONUCO)', 'value': 'PURACE (COCONUCO)'},
                                    {'label': 'SILVIA', 'value': 'SILVIA'},
                                    {'label': 'BOLIVAR', 'value': 'BOLIVAR'},
                                    {'label': 'PATÍA', 'value': 'PATÍA'},
                                    {'label': 'CALOTO', 'value': 'CALOTO'},
                                    {'label': 'TIMBÍO', 'value': 'TIMBÍO'},
                                    {'label': 'PIENDAMÓ - TUNÍA', 'value': 'PIENDAMÓ - TUNÍA'},
                                    {'label': 'GUAPÍ', 'value': 'GUAPÍ'},
                                    {'label': 'SUAREZ', 'value': 'SUAREZ'},
                                    {'label': 'CALDONO', 'value': 'CALDONO'},
                                    {'label': 'PATIA(EL BORDO)', 'value': 'PATIA(EL BORDO)'},
                                    {'label': 'INZA', 'value': 'INZA'},
                                    {'label': 'ROSAS', 'value': 'ROSAS'},
                                    {'label': 'BALBOA', 'value': 'BALBOA'},
                                    {'label': 'PUERTO TEJADA', 'value': 'PUERTO TEJADA'},
                                    {'label': 'BOLÍVAR', 'value': 'BOLÍVAR'},
                                    {'label': 'CAJIBIO', 'value': 'CAJIBIO'},
                                    {'label': 'TIMBIO', 'value': 'TIMBIO'},
                                    {'label': 'PIENDAMO', 'value': 'PIENDAMO'},
                                    {'label': 'LÓPEZ DE MICAY', 'value': 'LÓPEZ DE MICAY'},
                                    {'label': 'TIMBIQUI', 'value': 'TIMBIQUI'},
                                    {'label': 'PADILLA', 'value': 'PADILLA'},
                                    {'label': 'SAN SEBASTIAN', 'value': 'SAN SEBASTIAN'},
                                    {'label': 'JAMBALO', 'value': 'JAMBALO'},
                                    {'label': 'CAJIBÍO', 'value': 'CAJIBÍO'},
                                    {'label': 'SUÁREZ', 'value': 'SUÁREZ'},
                                    {'label': 'TOTORÓ', 'value': 'TOTORÓ'},
                                    {'label': 'GUAPI', 'value': 'GUAPI'},
                                    {'label': 'FLORENCIA', 'value': 'FLORENCIA'},
                                    {'label': 'SOTARA (PAISPAMBA)', 'value': 'SOTARA (PAISPAMBA)'},
                                    {'label': 'SOTARA', 'value': 'SOTARA'},
                                    {'label': 'SAN SEBASTIÁN', 'value': 'SAN SEBASTIÁN'},
                                    {'label': 'PÁEZ', 'value': 'PÁEZ'},
                                    {'label': 'TOTORO', 'value': 'TOTORO'},
                                    {'label': 'TIMBIQUÍ', 'value': 'TIMBIQUÍ'},
                                    {'label': 'SANTA ROSA', 'value': 'SANTA ROSA'},
                                    {'label': 'PIAMONTE', 'value': 'PIAMONTE'}
                                ],
                                placeholder='Seleccione un municipio',
                                className='custom-dropdown'
                            ),
                            html.Label('COLE_NATURALEZA:'),
                                dcc.Dropdown(
                                    id='input_cole_naturaleza',
                                    options=[
                                        {'label': 'OFICIAL', 'value': 'OFICIAL'},
                                        {'label': 'NO OFICIAL', 'value': 'NO OFICIAL'}
                                    ],
                                    placeholder='Seleccione la naturaleza del colegio',
                                    className='custom-dropdown'
                                ),
                            html.Label('ESTU_DEPTO_RESIDE:'),
dcc.Dropdown(
    id='input_estu_depto_reside',
    options=[
        {'label': 'CAUCA', 'value': 'CAUCA'},
        {'label': 'VALLE', 'value': 'VALLE'},
        {'label': 'NARIÑO', 'value': 'NARIÑO'},
        {'label': 'HUILA', 'value': 'HUILA'},
        {'label': 'BOGOTÁ', 'value': 'BOGOTÁ'},
        {'label': 'PUTUMAYO', 'value': 'PUTUMAYO'},
        {'label': 'TOLIMA', 'value': 'TOLIMA'},
        {'label': 'BOYACA', 'value': 'BOYACA'},
        {'label': 'CALDAS', 'value': 'CALDAS'},
        {'label': 'BOLIVAR', 'value': 'BOLIVAR'},
        {'label': 'CUNDINAMARCA', 'value': 'CUNDINAMARCA'},
        {'label': 'CAQUETA', 'value': 'CAQUETA'},
        {'label': 'META', 'value': 'META'},
        {'label': 'CESAR', 'value': 'CESAR'},
        {'label': 'ANTIOQUIA', 'value': 'ANTIOQUIA'},
        {'label': 'RISARALDA', 'value': 'RISARALDA'},
        {'label': 'CHOCO', 'value': 'CHOCO'},
        {'label': 'SAN ANDRES', 'value': 'SAN ANDRES'},
        {'label': 'ATLANTICO', 'value': 'ATLANTICO'},
        {'label': 'QUINDIO', 'value': 'QUINDIO'},
        {'label': 'SANTANDER', 'value': 'SANTANDER'}
    ],
    placeholder='Seleccione el departamento de residencia',
    className='custom-dropdown'
),

                                html.Label('ESTU_GENERO:'),
                                    dcc.Dropdown(
                                        id='input_estu_genero',
                                        options=[
                                            {'label': 'Femenino', 'value': 'F'},
                                            {'label': 'Masculino', 'value': 'M'}
                                        ],
                                        placeholder='Seleccione el género del estudiante',
                                        className='custom-dropdown'
                                    ),
                                html.Label('FAMI_EDUCACIONMADRE:'),
                                    dcc.Dropdown(
                                        id='input_fami_educacionmadre',
                                        options=[
                                            {'label': 'Educación profesional completa', 'value': 'Educación profesional completa'},
                                            {'label': 'Primaria completa', 'value': 'Primaria completa'},
                                            {'label': 'Secundaria (Bachillerato) completa', 'value': 'Secundaria (Bachillerato) completa'},
                                            {'label': 'Primaria incompleta', 'value': 'Primaria incompleta'},
                                            {'label': 'Postgrado', 'value': 'Postgrado'},
                                            {'label': 'No sabe', 'value': 'No sabe'},
                                            {'label': 'Secundaria (Bachillerato) incompleta', 'value': 'Secundaria (Bachillerato) incompleta'},
                                            {'label': 'Ninguno', 'value': 'Ninguno'},
                                            {'label': 'Educación profesional incompleta', 'value': 'Educación profesional incompleta'},
                                            {'label': 'Técnica o tecnológica completa', 'value': 'Técnica o tecnológica completa'},
                                            {'label': 'Técnica o tecnológica incompleta', 'value': 'Técnica o tecnológica incompleta'},
                                            {'label': 'No Aplica', 'value': 'No Aplica'}
                                        ],
                                        placeholder='Seleccione el nivel educativo de la madre',
                                        className='custom-dropdown'
                                    ),
                                    html.Label('FAMI_EDUCACIONPADRE:'),
                                    dcc.Dropdown(
                                        id='input_fami_educacionpadre',
                                        options=[
                                            {'label': 'Educación profesional completa', 'value': 'Educación profesional completa'},
                                            {'label': 'Primaria completa', 'value': 'Primaria completa'},
                                            {'label': 'Secundaria (Bachillerato) completa', 'value': 'Secundaria (Bachillerato) completa'},
                                            {'label': 'Primaria incompleta', 'value': 'Primaria incompleta'},
                                            {'label': 'Postgrado', 'value': 'Postgrado'},
                                            {'label': 'No sabe', 'value': 'No sabe'},
                                            {'label': 'Secundaria (Bachillerato) incompleta', 'value': 'Secundaria (Bachillerato) incompleta'},
                                            {'label': 'Ninguno', 'value': 'Ninguno'},
                                            {'label': 'Educación profesional incompleta', 'value': 'Educación profesional incompleta'},
                                            {'label': 'Técnica o tecnológica completa', 'value': 'Técnica o tecnológica completa'},
                                            {'label': 'Técnica o tecnológica incompleta', 'value': 'Técnica o tecnológica incompleta'},
                                            {'label': 'No Aplica', 'value': 'No Aplica'}
                                        ],
                                        placeholder='Seleccione el nivel educativo de la madre',
                                        className='custom-dropdown'
                                    ),
                                    html.Label('FAMI_ESTRATOVIVIENDA:'),
                                        dcc.Dropdown(
                                            id='input_fami_estratovivienda',
                                            options=[
                                                {'label': 'Estrato 1', 'value': 'Estrato 1'},
                                                {'label': 'Estrato 2', 'value': 'Estrato 2'},
                                                {'label': 'Estrato 3', 'value': 'Estrato 3'},
                                                {'label': 'Estrato 4', 'value': 'Estrato 4'},
                                                {'label': 'Estrato 5', 'value': 'Estrato 5'},
                                                {'label': 'Estrato 6', 'value': 'Estrato 6'},
                                                {'label': 'Sin Estrato', 'value': 'Sin Estrato'}
                                            ],
                                            placeholder='Seleccione el estrato de vivienda',
                                            className='custom-dropdown'
                                        ),
                                        html.Label('FAMI_TIENECOMPUTADOR:'),
                                        dcc.Dropdown(
                                            id='input_fami_tienecomputador',
                                            options=[
                                                {'label': 'Sí', 'value': 'Si'},
                                                {'label': 'No', 'value': 'No'}
                                            ],
                                            placeholder='¿Tiene computador?',
                                            className='custom-dropdown'
                                        ),
                                        html.Label('FAMI_TIENEINTERNET:'),
                                            dcc.Dropdown(
                                                id='input_fami_tieneinternet',
                                                options=[
                                                    {'label': 'Sí', 'value': 'Si'},
                                                    {'label': 'No', 'value': 'No'}
                                                ],
                                                placeholder='¿Tiene internet?',
                                                className='custom-dropdown'
                                            )


                        ], style={'marginBottom': '20px'}),

                        html.Button('Predecir', id='btn_predecir', n_clicks=0, className='custom-button'),
                        html.Div(id='output_prediccion', style={'marginTop': '20px', 'fontSize': '45px'})
                    ])
                ])

            ]
        )
    ]
)

# Callbacks para actualizar gráficos de Colegios
@app.callback(
    [Output('graph_colegios_bar', 'figure'),
     Output('graph_colegios_box', 'figure')],
    [Input('dropdown_colegios', 'value')]
)
def actualizar_colegios(columna):
    df_cleaned = datos.dropna(subset=[columna, 'PUNT_GLOBAL'])
    category_avg_score = df_cleaned.groupby(columna)['PUNT_GLOBAL'].mean().reset_index()
    fig_bar = px.bar(
        category_avg_score,
        x=columna,
        y='PUNT_GLOBAL',
        title=f'Promedio de Puntaje Global por {columna}',
        text='PUNT_GLOBAL',
        color=columna,
        labels={columna: 'Categoría', 'PUNT_GLOBAL': 'Promedio de Puntaje Global'},
        template='plotly_dark'
    )
    fig_box = px.box(
        df_cleaned,
        x=columna,
        y='PUNT_GLOBAL',
        color=columna,
        title=f'Distribución de Puntaje Global por {columna}',
        labels={columna: 'Categoría', 'PUNT_GLOBAL': 'Puntaje Global'},
        template='plotly_dark'
    )
    return fig_bar, fig_box

# Callbacks para actualizar gráficos de Competencias
@app.callback(
    [Output('graph_competencias_1', 'figure'),
     Output('graph_competencias_2', 'figure'),
     Output('graph_competencias_3', 'figure')],
    [Input('dropdown_competencias', 'value')]
)
def actualizar_competencias(competencia):
    fig1 = px.scatter(
        datos,
        x=competencia,
        y='PUNT_GLOBAL',
        title=f'Relación entre {competencia} y Puntaje Global',
        labels={'PUNT_GLOBAL': 'Puntaje Global', competencia: f'Puntaje en {competencia}'},
        template='plotly_dark'
    )
    stats = datos[competencia].describe().round(2)
    stats_df = pd.DataFrame({
        'Estadística': ['mean', 'std', 'min', '25%', '50%', '75%', 'max'],
        'Valor': stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']].values
    })
    fig2 = px.bar(
        stats_df,
        x='Estadística',
        y='Valor',
        title=f'Estadísticas Descriptivas de {competencia}',
        labels={'Valor': 'Valores', 'Estadística': 'Estadísticas'},
        text='Valor',
        template='plotly_dark'
    )
    fig3 = px.histogram(
        datos,
        x=competencia,
        nbins=20,
        title=f'Distribución de {competencia}',
        labels={competencia: f'Puntaje en {competencia}'},
        marginal='box',
        opacity=0.7,
        template='plotly_dark'
    )
    return fig1, fig2, fig3

# Callbacks para actualizar gráficos de Socioeconómico
@app.callback(
    [Output('graph_socioeconomico_bar', 'figure'),
     Output('graph_socioeconomico_box', 'figure')],
    [Input('dropdown_socioeconomico', 'value')]
)
def actualizar_socioeconomico(columna):
    df_cleaned = datos.dropna(subset=[columna, 'PUNT_GLOBAL'])
    category_avg_score = df_cleaned.groupby(columna)['PUNT_GLOBAL'].mean().reset_index()
    fig_bar = px.bar(
        category_avg_score,
        x=columna,
        y='PUNT_GLOBAL',
        title=f'Promedio de Puntaje Global por {columna}',
        text='PUNT_GLOBAL',
        color=columna,
        labels={columna: 'Categoría', 'PUNT_GLOBAL': 'Promedio de Puntaje Global'},
        template='plotly_dark'
    )
    fig_box = px.box(
        df_cleaned,
        x=columna,
        y='PUNT_GLOBAL',
        color=columna,
        title=f'Distribución de Puntaje Global por {columna}',
        labels={columna: 'Categoría', 'PUNT_GLOBAL': 'Puntaje Global'},
        template='plotly_dark'
    )
    return fig_bar, fig_box

@app.callback(
    Output('output_prediccion', 'children'),  # Mostrar el resultado
    [Input('btn_predecir', 'n_clicks')],  # Evento del botón
    [
        Input('input_cole_area', 'value'),
        Input('input_cole_bilingue', 'value'),
        Input('input_cole_calendario', 'value'),
        Input('input_cole_caracter', 'value'),
        Input('input_cole_genero', 'value'),
        Input('input_cole_jornada', 'value'),
        Input('input_cole_mcpio', 'value'),
        Input('input_cole_naturaleza', 'value'),
        Input('input_estu_depto_reside', 'value'),
        Input('input_estu_genero', 'value'),
        Input('input_fami_educacionmadre', 'value'),
        Input('input_fami_estratovivienda', 'value'),
        Input('input_fami_tienecomputador', 'value'),
        Input('input_fami_tieneinternet', 'value')
    ]
)
def predecir_puntaje_dash(n_clicks, cole_area, cole_bilingue, cole_calendario, cole_caracter, cole_genero,
                          cole_jornada, cole_mcpio, cole_naturaleza, estu_depto, estu_genero,
                          fami_educacionmadre, fami_estratovivienda, fami_tienecomputador, fami_tieneinternet):
    if n_clicks > 0:  # Solo ejecuta si el botón fue presionado
        # Construir el diccionario de entrada
        input_data = {
            'COLE_AREA_UBICACION': cole_area,
            'COLE_BILINGUE': cole_bilingue,
            'COLE_CALENDARIO': cole_calendario,
            'COLE_CARACTER': cole_caracter,
            'COLE_GENERO': cole_genero,
            'COLE_JORNADA': cole_jornada,
            'COLE_MCPIO_UBICACION': cole_mcpio,
            'COLE_NATURALEZA': cole_naturaleza,
            'ESTU_DEPTO_RESIDE': estu_depto,
            'ESTU_GENERO': estu_genero,
            'FAMI_EDUCACIONMADRE': fami_educacionmadre,
            'FAMI_ESTRATOVIVIENDA': fami_estratovivienda,
            'FAMI_TIENECOMPUTADOR': fami_tienecomputador,
            'FAMI_TIENEINTERNET': fami_tieneinternet
        }

        try:
            # Usar la función predecir_puntaje para calcular el puntaje predicho
            puntaje_predicho = predecir_puntaje(input_data)

            # Mostrar el resultado al usuario
            return f"Puntaje Global Predicho: {puntaje_predicho:.2f}"
        except Exception as e:
            # Manejo de errores si ocurre un problema con la predicción
            return f"Error en la predicción: {e}"
    return "Ingrese los valores y haga clic en Predecir."

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
