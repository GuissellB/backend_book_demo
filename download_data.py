import boto3
import os


def download_from_s3(
        bucket_name: str,
        s3_prefix: str,
        local_dir: str
):
    session = boto3.Session()

    s3 = session.resource('s3')
    bucket = s3.Bucket(bucket_name)

    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    print(
        f"Descargando datos de s3://{bucket_name}/{s3_prefix} a {local_dir}"
        f"..."
    )

    for obj in bucket.objects.filter(Prefix=s3_prefix):
        if obj.key.endswith('/'):
            continue

        target_path = os.path.join(
            local_dir,
            os.path.relpath(obj.key, s3_prefix)
        )
        target_folder = os.path.dirname(target_path)

        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        print(f"Descargando {obj.key}...")
        bucket.download_file(obj.key, target_path)

    print("Descarga completa.")


if __name__ == "__main__":
    BUCKET_NAME = "proyecto-guiss-b"
    S3_PREFIX = "data"  # en caso que necesites descargar el archivo específico
    LOCAL_DIR = "./"  # directorio local donde estarán los datos

    download_from_s3(BUCKET_NAME, S3_PREFIX, LOCAL_DIR)