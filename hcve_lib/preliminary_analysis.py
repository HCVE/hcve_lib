from pandas import DataFrame
import plotly.graph_objects as go


def plot_association_matrix(X_assocation: DataFrame) -> None:
    X_assocation = X_assocation[::-1]
    fig = go.Figure(
        data=go.Heatmap(
            z=X_assocation.values,
            x=X_assocation.columns,
            y=X_assocation.index,
            colorscale='balance',
            zmin=-1,
            zmax=1,
        )
    )

    # Customize the layout
    fig.update_layout(
        title='Correlation Matrix',
        width=1000,
        height=1000,
        autosize=False,
        xaxis=dict(
            title='Variables',
            side='top',
            tickmode='array',
            tickvals=list(range(len(X_assocation.columns))),
            ticktext=X_assocation.columns,
            tickangle=-45,
            automargin=True
        ),
        yaxis=dict(
            title='Variables',
            tickmode='array',
            tickvals=list(range(len(X_assocation.index))),
            ticktext=X_assocation.index,
            automargin=True
        ),
        plot_bgcolor='white'
    )
    fig.show()
