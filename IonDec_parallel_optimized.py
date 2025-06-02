
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 10:16:50 2025

@author: mvcalcism
"""


#####----- PRUEBA DEL MODELO DE DECONVOLUCIÓN DE ESPECTROS DIA -----#####


####----- Librerías necesarias -----####

import numpy as np
import pandas as pd
import pyopenms
from pyopenms import MSSpectrum, Precursor, MSExperiment, MzMLFile
import concurrent.futures
import argparse
import os
from tqdm import tqdm
import logging
import sys
import math









def IonDec_processing(exp):

# for number_scan, spectrum in tqdm(enumerate(exp, start = 1)):
    
    n_scans = []
    n_charges = []
    n_rt = []
    n_pepmass = []
    n_mz = []
    
    swath = 24

    # Extraer información del scan de interés.
    number_scan = exp.SCANS
    
    rt_id = exp.RT

    mquery = exp.MZ_PRECURSOR

    fr = pd.DataFrame(zip(exp.MZ_FRAGMENT, exp.INT_FRAGMENT))
    
    fr = fr.iloc[np.where(fr.iloc[:,1] != 0)[0],:]
    

    ########## for x in (index_rawfile[1,index_tabla]:index_rawfile[2,index_tabla]):
    
    # Separación de columnas. 
    fr_mass = fr.iloc[:,0]
    fr_int = fr.iloc[:,1]
    
    # Separar parte entera y decimal. 
    floor_num = np.floor(fr_mass).astype(int)
    fr_tmp = np.column_stack((floor_num, fr_mass, fr_int))
    fr_tmp = pd.DataFrame(fr_tmp)
    
    # Duplicación de datos. 
    fr_ch1 = fr_tmp.copy()
    fr_ch2 = fr_tmp.copy()
    
    
    # Filtrado por masas carga 1. 
    df_ch1 = fr_ch1.copy()
    df_ch1.columns = ["pos", "val","intens"]
    
    # Error. 
    df_ch1["error"] = (50 * df_ch1["pos"]) / 1_000_000
    
    # Rangos para cada función (vectorizado). 
    def get_bounds(df, factor, offset):
        fun = factor * df_ch1["pos"] + offset
        upper = df_ch1["pos"] + fun + df_ch1["error"]
        lower = df_ch1["pos"] + fun - df_ch1["error"]
        return lower, upper
    
    # Rangos de las 3 funciones. 
    low_nm, up_nm = get_bounds(df_ch1, 0.00049, 0.0418)
    low_1phos, up_1phos = get_bounds(df_ch1, 0.0005, -0.0401)
    low_2phos, up_2phos = get_bounds(df_ch1, 0.0005, -0.1133)
    
    # Comprobamos si está en al menos uno de los rangos. 
    mask_nm = (df_ch1["val"] >= low_nm) & (df_ch1["val"] <= up_nm)
    mask_1phos = (df_ch1["val"] >= low_1phos) & (df_ch1["val"] <= up_1phos)
    mask_2phos = (df_ch1["val"] >= low_2phos) & (df_ch1["val"] <= up_2phos)
    
    # Unimos. 
    mask_keep = mask_nm | mask_1phos | mask_2phos
    
    # Filtramos. 
    fr_ch1 = df_ch1[mask_keep].copy()
    fr_ch1 = fr_ch1[["val", "intens"]]
    fr_ch1.columns = ["mass", "int"]

    
    
    
    # Filtrado por masas carga 2. 
    df_ch2 = fr_ch2.copy()
    df_ch2.columns = ["pos", "val","intens"]
    
    # Error. 
    df_ch2["error"] = (50 * df_ch2["pos"]) / 1_000_000
    
    # Rangos para cada función (vectorizado). 
    def get_bounds(df, factor, offset):
        fun = factor * df_ch2["pos"] + offset
        upper = df_ch2["pos"] + fun + df_ch2["error"]
        lower = df_ch2["pos"] + fun - df_ch2["error"]
        return lower, upper
    
    # Rangos de las 3 funciones. 
    low_nm, up_nm = get_bounds(df_ch2, 0.0005, 0.5244)
    low_1phos, up_1phos = get_bounds(df_ch2, 0.0005, 0.4837)
    low_2phos, up_2phos = get_bounds(df_ch2, 0.0005, 0.4472)
    
    # Comprobamos si está en al menos uno de los rangos. 
    mask_nm = (df_ch2["val"] >= low_nm) & (df_ch2["val"] <= up_nm)
    mask_1phos = (df_ch2["val"] >= low_1phos) & (df_ch2["val"] <= up_1phos)
    mask_2phos = (df_ch2["val"] >= low_2phos) & (df_ch2["val"] <= up_2phos)
    
    # Unimos. 
    mask_keep = mask_nm | mask_1phos | mask_2phos
    
    # Filtramos. 
    fr_ch2 = df_ch2[mask_keep].copy()
    fr_ch2 = fr_ch2[["val", "intens"]]
    fr_ch2.columns = ["mass", "int"]
    
            
    

    # Filtrar. 
    fr_ch1 = fr_ch1[fr_ch1["mass"] > 180].reset_index(drop = True)
    fr_ch2 = fr_ch2[fr_ch2["mass"] > 180].reset_index(drop = True)
    fr_nored = pd.concat([fr_ch1, fr_ch2], ignore_index = True)
    fr_nored = fr_nored.sort_values(by = "mass").reset_index(drop = True)
    
    # Extraer masas e intensidades. 
    mass_ch1, int_ch1 = fr_ch1["mass"].values, fr_ch1["int"].values
    mass_ch2, int_ch2 = fr_ch2["mass"].values, fr_ch2["int"].values
    
    
    
    
    if fr_nored.shape[0] > 50:
    
    
        ####----- Preparación de la matriz de sumas y restas, e intensidades (CARGA +1) -----####
        
        # Crear la matriz cuadrada. 
        fmass_ch1 = np.tile(mass_ch1, (len(mass_ch1), 1))
        tfmass_ch1 = fmass_ch1.T
        
        # Matrices de suma y resta de masas. 
        sum_matrix_ch1 = np.abs(fmass_ch1 + tfmass_ch1 - 1.0078)
        diff_matrix_ch1 = np.abs(fmass_ch1 - tfmass_ch1)
        
        # Crear IDs de fila. 
        names_sum_matrix = list(map(str, range(1, sum_matrix_ch1.shape[0] + 1)))
        
        # Convertir a df y asignar índices. 
        df_sum_matrix = pd.DataFrame(sum_matrix_ch1, index = names_sum_matrix, columns = names_sum_matrix)
        df_diff_matrix = pd.DataFrame(diff_matrix_ch1, index = names_sum_matrix, columns = names_sum_matrix)
        
        # Matriz de suma de intensidades. 
        fint_ch1 = np.tile(int_ch1, (len(int_ch1), 1))
        tfint_ch1 = fint_ch1.T
        sum_matrix2_ch1 = np.abs(fint_ch1 + tfint_ch1)
        
        df_sum_matrix2 = pd.DataFrame(sum_matrix2_ch1, index = names_sum_matrix, columns = names_sum_matrix)
        
        # Reordenación inversa y transposición.
        rev_sum_matrix = df_sum_matrix.loc[::-1].T
        rev_diff_matrix = df_diff_matrix.loc[::-1].T
        rev_sum_matrix2 = df_sum_matrix2.loc[::-1].T
        
        # Convertir matrices a formato largo. 
        rev_sum_matrix_long = rev_sum_matrix.reset_index().melt(id_vars = 'index', var_name = 'income', value_name = 'value')
        rev_sum_matrix2_long = rev_sum_matrix2.reset_index().melt(id_vars = 'index', var_name = 'income', value_name = 'value')
        rev_diff_matrix_long = rev_diff_matrix.reset_index().melt(id_vars = 'index', var_name = 'income', value_name = 'value')
        
        # Combinar las columnas. 
        mass_int_ch1 = pd.DataFrame({
            'sum_mass': rev_sum_matrix_long['value'],
            'sum_int': rev_sum_matrix2_long['value'],
            'diff_mass': rev_diff_matrix_long['value']
        })
        

        
        # Filtro según las ventanas de masa para carga 2. 
        mass_int_pepmass1 = mass_int_ch1[
            (mass_int_ch1.iloc[:, 0] > float(mquery) * 2 - swath) &
            (mass_int_ch1.iloc[:, 0] < float(mquery) * 2 + swath)
        ]

        # Elimina filas duplicadas. 
        mass_int_pepmass1 = mass_int_pepmass1.drop_duplicates().reset_index(drop = True)
        


        
        ####----- Preparación de la matriz de sumas y restas, e intensidades (CARGA +2) -----####
        
        # Matriz de suma de masas carga 3. 
        fmass_ch2 = np.tile(mass_ch2, (len(mass_ch1), 1))
        tfmass_ch2 = np.tile(mass_ch1.reshape(-1, 1), (1, len(mass_ch2)))
        sum_matrix_ch2 = np.abs(fmass_ch2 * 2 + tfmass_ch2 - 1.0078 * 2)
        
        # Matriz de diferencia de masas. 
        diff_matrix_ch2 = np.abs(fmass_ch2 - tfmass_ch2)
        
        # Identificadores para las filas. 
        names_sum_matrix = list(map(str, range(1, sum_matrix_ch2.shape[0] + 1)))
        
        # Convierte las matrices de sumas y restas a df con los índices. 
        df_sum_matrix = pd.DataFrame(sum_matrix_ch2, index = names_sum_matrix)
        df_diff_matrix = pd.DataFrame(diff_matrix_ch2, index = names_sum_matrix)
        
        # Matriz de suma de intensidades. 
        fint_ch2 = np.tile(int_ch2, (len(int_ch1), 1))
        tfint_ch2 = np.tile(int_ch1.reshape(-1, 1), (1, len(int_ch2)))
        sum_matrix2_ch2 = np.abs(fint_ch2 + tfint_ch2)
        
        # Convierte a df y asigna nombres de filas. 
        df_sum_matrix2 = pd.DataFrame(sum_matrix2_ch2, index=names_sum_matrix)
        
        # Reordena las matrices invirtiendo el orden de las filas y transponiendo. 
        rev_sum_matrix = df_sum_matrix.loc[::-1].T
        rev_diff_matrix = df_diff_matrix.loc[::-1].T
        rev_sum_matrix2 = df_sum_matrix2.loc[::-1].T
        
        # Convierte cada matriz en formato long. 
        rev_sum_matrix = pd.melt(rev_sum_matrix.reset_index(), id_vars = 'index', var_name = 'income')
        rev_diff_matrix = pd.melt(rev_diff_matrix.reset_index(), id_vars = 'index', var_name = 'income')
        rev_sum_matrix2 = pd.melt(rev_sum_matrix2.reset_index(), id_vars = 'index', var_name = 'income')
        
        # Combina las columnas de sumas de masas, intensidades y diferencias en un solo df. 
        mass_int_ch2 = pd.DataFrame({
            'sum_mass': rev_sum_matrix['value'],
            'sum_int': rev_sum_matrix2['value'],
            'diff_mass': rev_diff_matrix['value']
        })
        


        # Filtro según las ventanas de masa para carga 3. 
        mass_int_pepmass2 = mass_int_ch2[
            (mass_int_ch2.iloc[:, 0] > float(mquery) * 3 - swath) &
            (mass_int_ch2.iloc[:, 0] < float(mquery) * 3 + swath)
        ]

        # Elimina filas duplicadas. 
        mass_int_pepmass2 = mass_int_pepmass2.drop_duplicates().reset_index(drop = True)
        


        # Unión de ambas ventanas. 
        mass_int = pd.concat([mass_int_pepmass1, mass_int_pepmass2], ignore_index = True)

        
        # Filtra filas con diferencia de masa mayor que 0. 
        mass_int = mass_int[mass_int.iloc[:, 2] > 0]
        
        # Renombra las columnas. 
        mass_int.columns = ["mass_sum", "int", "mass_diff"]
        
        # Ordena por la suma de masas en orden creciente. 
        mass_int = mass_int.sort_values(by = "mass_sum", ascending = True).reset_index(drop = True)
        
        
        
        
        if mass_int.shape[0] > 10:
        
        
            ####----- Agrupación de picos y diferencia de sumas -----####
            
            # Copia el df original. 
            red_mass_int = mass_int.copy()
            
            # Ordena las intensidades en orden descendente. 
            red_mass_int = red_mass_int.sort_values(by = "int", ascending = False).reset_index(drop = True)
            
            # Sensibilidad. 
            i_sens = 0.007
            
            # Suma total de intensidades. 
            sum_int_total = red_mass_int["int"].sum()
            
            # Índice. 
            i = 0
            
            # Bucle para agrupar intensidades similares por masa. 
            while i < len(red_mass_int):
                n_sup = red_mass_int.loc[i, "mass_sum"] + i_sens
                n_inf = red_mass_int.loc[i, "mass_sum"] - i_sens
                
                # Ííndices donde la suma de masas está dentro del rango. 
                n_pos = red_mass_int[
                    (red_mass_int["mass_sum"] < n_sup) &
                    (red_mass_int["mass_sum"] > n_inf)
                ].index.tolist()
                
                if len(n_pos) > 1:
                    int_act = red_mass_int.loc[n_pos, "int"]
                    sum_int = int_act.sum()
                    red_mass_int.at[i, "int"] = sum_int
                    red_mass_int.at[i, "count"] = len(n_pos)
                    
                    rel_int_gr = (sum_int / sum_int_total) * 100
                    red_mass_int.at[i, "rel_int"] = rel_int_gr
                    red_mass_int.at[i, "factor"] = rel_int_gr * len(n_pos)
                    
                    # Elimina las posiciones excepto la primera. 
                    n_pos_rest = [x for x in n_pos if x != i]
                    red_mass_int = red_mass_int.drop(index = n_pos_rest).reset_index(drop = True)
                    
                    i += 1
                else:
                    int_act = red_mass_int.loc[n_pos, "int"]
                    sum_int = int_act.sum()
                    red_mass_int.at[i, "int"] = sum_int
                    red_mass_int.at[i, "count"] = len(n_pos)
                    
                    rel_int_gr = (sum_int / sum_int_total) * 100
                    red_mass_int.at[i, "rel_int"] = rel_int_gr
                    red_mass_int.at[i, "factor"] = rel_int_gr * len(n_pos)
                    i += 1
            
            # Identifica posiciones donde no se calculó el número de picos. 
            if red_mass_int["count"].isna().any():
                na_idx = red_mass_int[red_mass_int["count"].isna()].index
                red_mass_int.loc[na_idx, "count"] = 1
            
                for e in na_idx:
                    rel_int_gr = (red_mass_int.at[e, "int"] / sum_int_total) * 100
                    red_mass_int.at[e, "rel_int"] = rel_int_gr
                    red_mass_int.at[e, "factor"] = rel_int_gr * 1
            
            # Ordena de nuevo por la suma de masas en orden creciente. 
            red_mass_int_m2 = red_mass_int.sort_values(by = "mass_sum", ascending = True).reset_index(drop = True)
            
            
            
            
            
            if red_mass_int_m2.shape[0] > 10:
            
            
                # Calcula las diferencias de las sumas de las masas y selecciona unos valores detrminados en las diferencias. 
                # Ordena por intensidad descendente. 
                red_mass_int_m2 = red_mass_int_m2.sort_values(by = "int", ascending = False).reset_index(drop = True)
                
                # Extrae la columna de sumas de masas. 
                mass_sums = red_mass_int_m2["mass_sum"].values
                
                # Crea una matriz cuadrada con sumas de masas. 
                sum_mass = np.tile(mass_sums, (len(mass_sums), 1))
                
                # Transpone la matriz. 
                t_sum_mass = sum_mass.T
                
                # Calcula la diferencia entre sumas de masas. 
                diff_sum_mass = sum_mass - t_sum_mass
                
                # Filtra las diferencias que estén en el rango -1.003, sino, se reemplazan con 0. 
                red_sum_mass = np.where(
                    (diff_sum_mass >= -1.0075) & (diff_sum_mass <= -0.9955),
                    diff_sum_mass,
                    0
                )
                
                red_sum_mass = pd.DataFrame(red_sum_mass)
                
                
                ####----- Cálculo de los precursores y las envolventes -----####
                
                lista_prec = []
                
                # Recorre cada fila de red_sum_mass. 
                for j in range(red_sum_mass.shape[0]):
                    tmp_vector = []
                    
                    # Si hay al menos una coincidencia. 
                    if np.sum(red_sum_mass.iloc[j,:]) != 0:
                        pos_prec = j
                        tmp_row = np.where(red_sum_mass.iloc[:,pos_prec] != 0)[0]
                        
                        if np.sum(red_sum_mass.iloc[:,pos_prec]) != 0:
                            tmp_val = red_mass_int_m2.iloc[pos_prec]["int"]
                            act_int = red_mass_int_m2.iloc[tmp_row]["int"].values
                            
                            # Calcula diferencia relativa entre intensidades. 
                            tmp_val2 = np.abs((tmp_val - act_int) / act_int)
                            pos_tmp = np.where(tmp_val2 == np.min(tmp_val2))[0]
                            tmp_row = tmp_row[pos_tmp]
                            
                            if len(tmp_row) > 1:
                                tmp_row = tmp_row[0].item()
                            else:
                                tmp_row = tmp_row.item()
                            
                            tmp_n = red_mass_int_m2.iloc[pos_prec]["int"]
                            tmp_a = red_mass_int_m2.iloc[tmp_row]["int"]
                            tmp_id = tmp_a - tmp_n * 0.75
                            
                            if tmp_id > 0:
                                tmp_vector.append(tmp_row)
                                
                        tmp_vector.append(pos_prec)
                        pos_row = pos_prec
                        
                        # Busca hacia adelante mientras haya coincidencias válidas. 
                        k = True
                        while k:
                            pos_col = np.where(red_sum_mass.iloc[pos_row,:] != 0)[0]
                            if len(pos_col) != 1:
                                tmp_val = red_mass_int_m2.iloc[pos_col]["int"].values
                                act_int = red_mass_int_m2.iloc[pos_row]["int"]
                                act_int = act_int.item()
                                tmp_val2 = np.abs((tmp_val - act_int) / act_int)
                                pos_col = pos_col[np.where(tmp_val2 == np.min(tmp_val2))[0]]
                                
                            if len(pos_col) > 1:
                                pos_col = pos_col[0].item()
                            else:
                                pos_col = pos_col.item()
                            
                            count_next = red_mass_int_m2.iloc[pos_col]["count"]
                            count_actual = red_mass_int_m2.iloc[pos_row]["count"]
                            id_prec_count = count_actual.item() - count_next.item() * 0.7
                
                            if id_prec_count > 0:
                                tmp_vector.append(pos_col)
                                pos_row = pos_col
                            else:
                                k = False
                            
                            if np.sum(red_sum_mass.iloc[pos_row,:]) == 0:
                                k = False
                
                        # Si encontró más de un punto, guarda los valores y elimina combinaciones usadas. 
                        if len(tmp_vector) > 1:
                            val_prec = red_mass_int_m2.iloc[tmp_vector]["mass_sum"].values
                            lista_prec.append(val_prec)
                            
                            for idx in tmp_vector:
                                red_sum_mass.iloc[idx,:] = 0
                                red_sum_mass.iloc[:,idx] = 0
                                
                # Elimina listas vacías. 
                lista_prec = [x for x in lista_prec if x is not None and len(x) > 0]
                
                
                
                
                if len(lista_prec) != 0:
                
                    
                
                    ####----- Búsqueda de posibles precursores con un solo pico -----####
                    
                    # Creamos una copia del df con posibles precursores restantes. 
                    s_prec_mat = red_mass_int_m2.copy()
                    
                    # Aplanamos la lista de listas de índices a un único array. 
                    n_pos_prec = np.concatenate(lista_prec).ravel()
                    
                    # Filtramos las filas que no están en los precursores ya identificados
                    s_prec_mat = s_prec_mat[~s_prec_mat["mass_sum"].isin(n_pos_prec)]
                    
                    # Filtramos las que tengan al menos 7 elementos en el grupo. 
                    s_prec_mat = s_prec_mat[s_prec_mat["count"] >= 7]
                    
                    # Nuevos posibles candidatos. 
                    if not s_prec_mat.empty:
                        for t in range(len(s_prec_mat)):
                            lista_prec.append([s_prec_mat.iloc[t]["mass_sum"]])
                    
                    # Limpieza. 
                    lista_prec = [x for x in lista_prec if x is not None and len(x) > 0]
                
                


                    ####----- Cálculo de los fragmentos de cada precursor (con suma intensidades) -----####
                    
                    prec_vec = np.array([x[0] for x in lista_prec])
                    sum_mat_frag_ch1 = np.triu(sum_matrix_ch1, k = 1)
                    sum_mat_frag_ch2 = sum_matrix_ch2
                    i_sens = 0.007
                    
                    # Información de los fragmentos. 
                    fr1_mass = fr_ch1.iloc[:, 0].astype(float).to_numpy()
                    fr1_int = fr_ch1.iloc[:, 1].astype(float).to_numpy()
                    fr2_mass = fr_ch2.iloc[:, 0].astype(float).to_numpy()
                    fr2_int = fr_ch2.iloc[:, 1].astype(float).to_numpy()
                    fr_nored_mass = fr_nored.iloc[:, 0].astype(float).to_numpy()
                    fr_nored_int = fr_nored.iloc[:, 1].astype(float).to_numpy()
                    
                    for g, prec_mass in enumerate(prec_vec):
                        scan_id = number_scan * 1000 + (g + 1)
                        mass_mh2 = round((prec_mass + 1.0078) / 2, 5)
                        n_sup = prec_mass + i_sens
                        n_inf = prec_mass - i_sens
                    
                        # Determina carga. 
                        in_swath = (mquery * 2 - swath) < prec_mass < (mquery * 2 + swath)
                        
                        if in_swath:
                            val_charge = 2
                            idx = np.where((sum_mat_frag_ch1 < n_sup) & (sum_mat_frag_ch1 > n_inf))
                            row_frag = fr1_mass[idx[0]]
                            row_int = fr1_int[idx[0]]
                            col_frag = fr1_mass[idx[1]]
                            col_int = fr1_int[idx[1]]
                        else:
                            val_charge = 3
                            idx = np.where((sum_mat_frag_ch2 < n_sup) & (sum_mat_frag_ch2 > n_inf))
                            row_frag = fr1_mass[idx[0]]
                            row_int = fr1_int[idx[0]]
                            col_frag = fr2_mass[idx[1]]
                            col_int = fr2_int[idx[1]]
                    
                        all_frags = np.concatenate([row_frag, col_frag])
                        all_int = np.concatenate([row_int, col_int])
                    
                        if len(all_frags) > 0:
                            mass_int_4 = np.column_stack((all_frags, all_int))
                    
                            # Búsqueda y1. 
                            ref_val = mass_int_4[-1, 0] if mass_int_4.shape[0] > 1 else prec_mass - 200
                        else:
                            ref_val = prec_mass - 200
                            mass_int_4 = np.empty((0, 2))
                    
                        pos_y1_high = np.where(fr_nored_mass > ref_val)[0]
                        pos_y1_low = np.where(fr_nored_mass < 180)[0]
                    
                        tmp_mayor = fr_nored_mass[pos_y1_high]
                        tmp_menor = fr_nored_mass[pos_y1_low]
                    
                        if tmp_mayor.size > 0 and tmp_menor.size > 0:
                            y1_vals_3 = np.tile(tmp_mayor, (len(tmp_menor), 1))
                            y1_vals_4 = np.tile(tmp_menor[:, None], (1, len(tmp_mayor)))
                    
                            if in_swath:
                                val_charge = 2
                                y1_sum = y1_vals_3 + y1_vals_4 - 1.0078
                            else:
                                val_charge = 3
                                y1_sum_2 = y1_vals_3 + y1_vals_4 * 2 - 2 * 1.0078
                                y1_sum_3 = y1_vals_3 * 2 + y1_vals_4 - 2 * 1.0078
                                y1_act = np.vstack([
                                    np.column_stack(np.where((y1_sum_2 < n_sup) & (y1_sum_2 > n_inf))),
                                    np.column_stack(np.where((y1_sum_3 < n_sup) & (y1_sum_3 > n_inf)))
                                ])
                            if in_swath:
                                y1_act = np.column_stack(np.where((y1_sum < n_sup) & (y1_sum > n_inf)))
                    
                            if y1_act.size > 0:
                                y1_low_vals = np.column_stack((fr_nored_mass[pos_y1_low[y1_act[:, 0]]],
                                                               fr_nored_int[pos_y1_low[y1_act[:, 0]]]))
                                y1_high_vals = np.column_stack((fr_nored_mass[pos_y1_high[y1_act[:, 1]]],
                                                                fr_nored_int[pos_y1_high[y1_act[:, 1]]]))
                                more_frags = np.vstack([y1_low_vals, y1_high_vals])
                                combined = np.vstack([mass_int_4, more_frags])
                    
                                # Elimina duplicados y ordena. 
                                mass_int_4 = np.unique(combined, axis = 0)
                                mass_int_4 = mass_int_4[np.argsort(mass_int_4[:, 0])]
                    
                        # Guarda si hay fragmentos. 
                        if mass_int_4.shape[0] > 0:
                            n_scans.append(scan_id)
                            n_charges.append(val_charge)
                            n_rt.append(rt_id)
                            n_pepmass.append(mass_mh2)
                            n_mz.append([tuple(row) for row in mass_int_4])

                    
    # Valores devueltos. 
    return([n_scans, n_charges, n_rt, n_pepmass, n_mz])
                
                
                    ########## }
                
                ########## }
                
            ########## }
            
        ########## }
        
    ########## }







def main(args):

    
    # Cargar el archivo mzML.
    logging.info("Reading mzML file...")
    exp = pyopenms.MSExperiment()
    file_name = "FE_x01439"
    infile = "C:/Users/mvcalcism/Desktop/TFM_P/Resultados_fosfoproteoma/" + file_name + ".mzML"
    
    # file_name = "Aortas_Marfan_TMT4_FR5"
    # infile = r"S:\U_Proteomica\LABS\LAB_JMR\Marfan\2024_EC_EN_DM_Tissue-cohorts\2024_DM_Human-Aortas_data\mzML\mzML\\" + file_name + ".mzML"
    
    pyopenms.MzMLFile().load(infile, exp)
    
    # Extraer todos los espectros y calcular el número. 
    logging.info("Extracting spectra...")    
    tquery = []
    for s in tqdm(exp.getSpectra()):
        if s.getMSLevel() == 2:
            mz_array, intensity_array = s.get_peaks()
            df = pd.DataFrame([int(s.getNativeID().split(' ')[-1][5:]), # Scan
                      s.getPrecursors()[0].getCharge(), # Precursor Charge
                      s.getRT(), # Precursor Retention Time
                      s.getPrecursors()[0].getMZ(), # Precursor MZ
                      list(mz_array),
                      list(intensity_array)])
            tquery.append(df.T)
    tquery = pd.concat(tquery)
    tquery.columns = ["SCANS", "CHARGE", "RT", "MZ_PRECURSOR", "MZ_FRAGMENT", "INT_FRAGMENT"]
    tquery.SCANS = tquery.SCANS.astype(int)
    tquery.CHARGE = tquery.CHARGE.astype(int)
    
    
    # Paralelizar. 
    indices, rowSeries = zip(*tquery.iterrows())
    rowSeries = list(rowSeries)
    tqdm.pandas(position = 0, leave = True)
    chunks = 100
    if len(tquery) <= chunks:
        chunks = math.ceil(len(tquery)/args.n_workers)
    logging.info("\tBatch size: " + str(chunks) + " (" + str(math.ceil(len(tquery)/chunks)) + " batches)")
    logging.info("Processing...")
    with concurrent.futures.ProcessPoolExecutor(max_workers = args.n_workers) as executor:
        result = list(tqdm(executor.map(IonDec_processing, rowSeries, chunksize = chunks), total = len(rowSeries)))



    # Inicializar listas.
    all_scans, all_charges, all_rt, all_pepmass, all_mz = [], [], [], [], []

    # Combinar los resultados de cada scan.
    for res in result:
        scans, charges, rt, pepmass, mz = res
        all_scans.extend(scans)
        all_charges.extend(charges)
        all_rt.extend(rt)
        all_pepmass.extend(pepmass)
        all_mz.extend(mz)


    logging.info("Creating corrected mzML file...")
    new_exp = MSExperiment()
    
    
    # Recorrer las listas acumuladas y crear un espectro para cada scan. 
    for val_scan, val_charge, val_rt, val_pepmass, val_frags in tqdm(zip(all_scans, all_charges, all_rt, all_pepmass, all_mz), total=len(all_scans)):
        spectrum = MSSpectrum()
    
        # Tiempo de retención. 
        spectrum.setRT(val_rt)
    
        # Nivel de MS. 
        spectrum.setMSLevel(2)
    
        # m/z e intensidad. 
        val_mz, val_int = zip(*val_frags)
        spectrum.set_peaks((list(val_mz), list(val_int)))
    
    
        # Pepmass. 
        precursor = Precursor()
        precursor.setMZ(val_pepmass)
        precursor.setCharge(val_charge)
        spectrum.setPrecursors([precursor])
    
        # Número de scan. 
        spectrum.setNativeID(f"scan={val_scan}")
    
        # Añadir el espectro al nuevo experimento. 
        new_exp.addSpectrum(spectrum)
    
    
    
    # Guardar el nuevo archivo mzML. 
    logging.info("Saving corrected mzML file...")
    outfile = "C:/Users/mvcalcism/Desktop/TFM_P/Resultados_fosfoproteoma/Resultados/" + file_name + "_IDed.mzML"
    MzMLFile().store(outfile, new_exp)





if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='IonDec',
        epilog='''
        Example:
            python IonDec.py
    
        ''')
            
    parser.add_argument('-w',  '--n_workers', type = int, default = os.cpu_count(), help = 'Number of threads/n_workers')

    args = parser.parse_args()
    
    logging.basicConfig(level = logging.INFO,
                        format = '%(asctime)s - %(levelname)s - %(message)s',
                        datefmt = '%m/%d/%Y %I:%M:%S %p',
                        handlers = [logging.StreamHandler()])
    logging.info('start script: '+"{0}".format(" ".join([x for x in sys.argv])))
    main(args)


