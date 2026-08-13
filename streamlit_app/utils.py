import plotly.express as px

def fraud_chart_style(fig):

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(color="white"),
        margin=dict(l=20,r=20,t=40,b=20),
        height=450
    )

    return fig