from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd

# Cargar los datos
file_path = 'datos_limpios.csv'  # Cambia esto si tu archivo está en otra ubicación
datos = pd.read_csv(file_path)

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

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
