from pathlib import Path
import requests
import rarfile


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def unpack_rar(source, destination=None):
    """Unpack a rar file
    """
    src_path = Path(source)
    if destination is None:
        dst_path = src_path.parent
    else:
        dst_path = Path(destination)

    rf = rarfile.RarFile(str(src_path))
    for f in rf.infolist():
        if f.isdir():
            continue

        dst_file_path = dst_path / f.filename
        print(str(dst_file_path))
        if not dst_file_path.parent.exists():
            dst_file_path.parent.mkdir(parents=True)
    
        with open(str(dst_file_path), 'wb') as unpacked_f:
            unpacked_f.write(rf.read(f))