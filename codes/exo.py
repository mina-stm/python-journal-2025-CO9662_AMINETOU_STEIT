# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 10:49:09 2026

@author: PC
"""

f = lambda x, y: x**2 + y


def e_potentielle(masse, hauteur,e_limite, g=9.81):
    E = masse * hauteur * g
    print((E < e_limite) | (E > e_limite ))
    print(E, 'Joules')
    return E
    
    
print('bonjour')
resultatex = e_potentielle(masse=80, hauteur=5, e_limite= 5555)


#listes
liste_1 =[1, 4, 2, 7, 35, 84]
villes = ['paris', 'berlin', 'londres', 'bruxelles']
liste_2 = [liste_1, villes]
liste_3 = []

#tuple inmodifiale
tuple_1 = (1,2,6,1,7)

#sting
prenom= 'mina'

print(villes[:3]) # 3 premier element
print(villes[1:3]) # de 1 a 3   element
print(villes[2:]) # apartir de 2 eme  element
print(villes[::2]) # de pas 2
print(villes[::-1]) # tous les element a l inverse

print(prenom[:3]) # 3 premier   element


traduction = {
    "chien":"dog",
    "chat":"cat",
    "souris":"mouse",
    "oiseau":"bird"
    }

inventaire = {
    "bananes":5000,
    "pommes":2094,
    "poires": 45555
    }

dict_3 = {
    "dict1":traduction,
    "dict2":inventaire
    }