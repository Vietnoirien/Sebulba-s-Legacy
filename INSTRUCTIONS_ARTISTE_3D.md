# Instructions pour l'Artiste 3D (Blender & Photoshop)

Ce document décrit les spécifications techniques pour garantir que les modèles 3D s'intègrent parfaitement dans notre simulation web (React Three Fiber).

## 1. Format de Fichier
Le format privilégié pour le web est le **.GLB** (glTF Binary).
*   **Pourquoi ?** Il regroupe le maillage, les matériaux et les textures en un seul fichier optimisé.
*   **Alternative** : **.OBJ** est accepté, mais vous devez fournir le fichier `.obj`, le fichier `.mtl` et toutes les textures dans un dossier compressé (ZIP).

## 2. Exportation depuis Blender

### Orientation et Axes
Le moteur de rendu web utilise un système de coordonnées spécifique. Merci de configurer l'export ainsi :
*   **Avant (Forward)** : **+X** (Axe X Positif). Si votre modèle regarde vers -Y ou +Z dans Blender, merci de le tourner de 90°.
*   **Haut (Up)** : **+Y** (Axe Y Positif).
*   **Geler les Transformations** : Assurez-vous d'appliquer toutes les transformations (Ctrl + A -> "All Transforms") avant l'export. L'échelle (Scale) doit être à `1, 1, 1` et la rotation à `0, 0, 0`.

### Échelle et Dimensions
*   L'unité de base dans Blender (1 mètre) correspond à 100 unités de jeu environ.
*   **Taille cible** : Le "Pod" (vaisseau) doit faire environ **4 mètres** de diamètre/longueur dans Blender.
*   **Point de Pivot (Origin)** : Centrez le point d'origine au centre de gravité du vaisseau (pas au niveau du sol, car ils flottent).

### Géométrie
*   **Polycount** : Visez moins de **5 000 triangles** par vaisseau si possible (low-poly / mid-poly).
*   **UVs** : Le modèle doit être impérativement déplié (UV Unwrapped). Pas d'UVs qui se chevauchent (sauf si intentionnel pour des motifs répétitifs).

### Paramètres d'Export glTF (Blender)
*   Format : `glTF Binary (.glb)`
*   Include : `Selected Objects` (coché)
*   Transform : `+Y Up` (coché)
*   Mesh : `Apply Modifiers` (coché), `UVs` (coché), `Normals` (coché)

## 3. Textures et Photoshop (PBR)

Nous utilisons un workflow **PBR** (Physically Based Rendering) standard.

### Cartes requises (Maps)
Veuillez fournir les textures suivantes :
1.  **Albedo / Base Color** : La couleur du vaisseau.
    *   *Note Importante* : Nous avons deux équipes (Rouge et Blanc). L'idéal est de fournir une texture **grise/blanche** neutre pour la carrosserie, afin que nous puissions la teinter dynamiquement via le code (Tint).
2.  **Normal Map** : Pour les détails de relief (Format OpenGL / Violet).
3.  **Metallic / Roughness** :
    *   Si vous exportez en `.glb`, Blender combine souvent ces cartes.
    *   Sinon, fournissez des maps en niveaux de gris (Blanc = Métallique/Rugueux, Noir = Non-métallique/Lisse).

### Résolution et Format
*   **Résolution** : **1024x1024** pixels (recommandé). 512x512 est acceptable. Évitez la 4K.
*   **Format** : **.PNG** (8-bit ou 16-bit) ou **.JPG** (qualité haute).

## 4. Résumé pour la Livraison
*   [ ] Fichier **.glb** (ou .obj + textures).
*   [ ] Modèle orienté vers **+X**.
*   [ ] Transformations appliquées (Scale 1, Rotation 0).
*   [ ] Textures fournies ou embarquées.
*   [ ] Point de pivot centré.
