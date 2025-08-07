import os
import subprocess

def merge_tifs_vrt_safe(root_dir, output_file, vrt_file='temp_merge.vrt', file_list='tif_list.txt'):
    tif_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith('.tif'):
                tif_files.append(os.path.join(dirpath, file))

    if not tif_files:
        print("Aucun fichier TIF trouvé dans le répertoire.")
        return

    print(f"Nombre de fichiers TIF trouvés : {len(tif_files)}")

    with open(file_list, 'w') as f:
        for tif in tif_files:
            f.write(tif + '\n')

    build_vrt_command = ["gdalbuildvrt", "-input_file_list", file_list, vrt_file]
    print(f"Construction du VRT avec la liste : {build_vrt_command}")
    subprocess.run(build_vrt_command, check=True)

    translate_command = ["gdal_translate", vrt_file, output_file]
    print(f"Fusion finale avec gdal_translate : {translate_command}")
    subprocess.run(translate_command, check=True)

    os.remove(vrt_file)
    os.remove(file_list)
    print(f"✅ Fichier fusionné créé : {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fusionner tous les fichiers TIF en un GeoTIFF départemental.')
    parser.add_argument('input_dir', type=str, help='Répertoire contenant les TIF à fusionner')
    parser.add_argument('output_tif', type=str, help='Chemin du fichier GeoTIFF fusionné de sortie')

    args = parser.parse_args()

    merge_tifs_vrt_safe(args.input_dir, args.output_tif)
