import cx_Oracle

# Datos Conexión a SIGMA_PROD_STB Energía

dsn_sg = cx_Oracle.makedsn("EPM-PO13", "1521", sid="GAGUPROD")

conn_sigma_aguas = cx_Oracle.connect(
    user="CONGAGUAS",
    password="CONGAGUAS1",
    dsn=dsn_sg
)