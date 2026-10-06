"""Módulo Satélites: diagnóstico remoto de um talhão a partir apenas do perímetro (KML/KMZ).

Frentes: histórico de uso (MapBiomas), comportamento da lavoura (Sentinel-2), relevo e água
(Copernicus DEM), perfil climático (CHIRPS/NASA POWER), contexto do solo (SoilGrids) e temperatura
de superfície (Landsat). Cada frente é independente: se uma fonte falhar, as demais seguem.
"""
