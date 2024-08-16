import os


def rename_images_in_folder(folder_path):
    # 获取指定文件夹中的所有文件
    files = os.listdir(folder_path)

    # 过滤出 PNG 文件
    png_files = [f for f in files if f.endswith('.png')]

    # 按照一定规则重命名文件
    for count, file_name in enumerate(png_files, start=1):
        # 构建新的文件名
        new_name = f'center_{count}.png'

        # 获取旧文件的完整路径
        old_file_path = os.path.join(folder_path, file_name)

        # 获取新文件的完整路径
        new_file_path = os.path.join(folder_path, new_name)

        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f'Renamed: {file_name} to {new_name}')


# 使用示例
folder_path = 'D:\\pworkspace\\RotateCaptcha\\temp'  # 替换为你的文件夹路径
rename_images_in_folder(folder_path)
