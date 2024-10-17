```python
from googleapiclient.discovery import build
from google.oauth2 import service_account
from flask import Flask, render_template


- **from googleapiclient.discovery import build** : Google API 클라이언트 라이브러리로, Google Drive API에 접근할 수 있도록 설정  
- **from google.oauth2 import service_account** : 서비스 계정을 사용하여 Google API에 인증할 때 필요한 라이브러리  
- **from flask import Flask, render_template** : 웹 프레임워크인 Flask를 사용하여 웹 서버를 구축  
- **render_template** : HTML 템플릿을 렌더링하여 Flask에서 동적 웹 페이지를 생성

```python
# JSON 파일 경로 (project2/credentials/JSON 파일)
SERVICE_ACCOUNT_FILE = 'credentials/my_api_key.json' # Google 서비스 계정으로 API 인증키를 통해 내려받은 json 파일

# Flask 앱 생성
app = Flask(__name__)

# Google Drive API 인증 설정
def authenticate_drive_api():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=['https://www.googleapis.com/auth/drive.readonly'] # Google API 페이지에서 읽기 권한으로 설정
    )
    service = build('drive', 'v3', credentials=creds)
    return service
```

- **SERVICE_ACCOUNT_FILE = 'credentials/my_api_key.json'** : Google Cloud에서 생성된 서비스 계정 JSON 파일 경로를 지정, 이 파일을 통해 Google Drive API에 접근할 수 있는 인증 정보를 제공  
- **app = Flask(__name__)** : Flask 애플리케이션 객체를 생성하여 app 변수에 인스턴스 할당. 이 객체를 통해 Flask의 기능을 사용 가능  
- **def authenticate_drive_api():** : Google Drive API에 인증하고 연결하기 위한 함수  
- **creds = service_account.Credentials.from_service_account_file(** : JSON 파일에서 서비스 계정 정보를 가져옴  
- **SERVICE_ACCOUNT_FILE,**  
- **scopes=['https://www.googleapis.com/auth/drive.readonly']** : API 호출 시 읽기 권한(drive.readonly)을 지정  
- **)**  
- **service = build('drive', 'v3', credentials=creds)** : Google Drive API를 호출할 수 있는 service 객체를 생성, 'drive'는 Drive API를 의미하고, 'v3'는 버전을 뜻함
- **return service**


```python
# Google Drive 특정 폴더의 파일 및 하위 폴더 가져오기
def get_drive_files(folder_id=None):
    service = authenticate_drive_api()
    
    # 폴더 ID가 없으면 기본적으로 root 폴더에서 시작
    if folder_id is None:
        folder_id = '연동하고자한는 구글 드라이브의 첫 번째 폴더의 링크에 표시되는 ID'
    
    # 폴더 안의 파일과 하위 폴더 가져오기
    results = service.files().list(
        q=f"'{folder_id}' in parents",
        pageSize=100,
        fields="files(id, name, mimeType, webViewLink)",
        orderBy="viewedByMeTime"
    ).execute()
    
    files = results.get('files', [])
    
    files_sorted = sorted(files, key=lambda x: (x['mimeType'] != 'application/vnd.google-apps.folder', x['name']))
    
    # 디버깅을 위해 가져온 파일을 출력
    print(files_sorted )
    
    return files_sorted     # 현재 열람하고자 하는 드라이브 폴더에 서비스 계정이 액세스되어 있어야 열람 가능
```

- **get_drive_files()** : 지정된 폴더 ID에 있는 파일과 하위 폴더를 가져오는 함수  
- **service = authenticate_drive_api():** : Google Drive API 인증을 받아 service 객체를 생성  
- **if folder_id is None:** : 폴더 ID를 지정하지 않으면 기본적으로 '연동하고자한는 구글 드라이브의 첫 번째 폴더의 링크에 표시되는 ID' (이것은 특정 폴더의 ID)로 설정됨  
- **results = service.files().list(** : Google Drive API를 통해 폴더 안의 파일 목록을 가져옴  
- **q=f"'{folder_id}' in parents",** : 검색 쿼리로, '{folder_id}' in parents는 해당 폴더 ID의 자식 파일을 모두 가져오라는 의미  
- **pageSize=100,** : 가져올 최대 파일 개수를 100개로 제한  
- **fields="files(id, name, mimeType, webViewLink)"** API 응답으로 파일의 id, name, mimeType (파일 형식), webViewLink (웹에서 볼 수 있는 링크)를 요청  
- **orderBy="viewedByMeTime"** : 파일 및 폴더의 정렬 기준 (createdTime: 파일이 생성된 날짜 기준으로 정렬, folder: 폴더 우선 정렬, modifiedTime: 수정된 시간 기준으로 정렬, name: 파일 이름 순으로 정렬, name_natural: 이름을 자연어 순으로 정렬, viewedByMeTime: 내가 마지막으로 본 시간 기준으로 정렬)  
 -**).execute()** : API 요청을 실행  
- **files = results.get('files', [])** : API에서 반환된 결과에서 files라는 키에 해당하는 파일 목록을 가져오고, 만약 파일이 없다면 빈 리스트를 반환  
- **files_sorted = sorted(files, key=lambda x: (x['mimeType'] != 'application/vnd.google-apps.folder', x['name']))** : 가져온 파일 목록을 정렬, key=lambda x: 정렬 기준을 지정하고, mimeType이 'application/vnd.google-apps.folder'(Google Drive 폴더)를 먼저 정렬하고, 파일 이름을 기준으로 정렬, 폴더는 먼저, 파일은 나중에, 그 다음에 파일 이름으로 알파벳 순서대로 정렬  
- **print(files_sorted)** : 출력하여 디버깅용으로 확인


```python
@app.route('/')
@app.route('/folder/<folder_id>')
def home(folder_id=None):
    files = get_drive_files(folder_id)
    return render_template('index.html', files=files)
```

- **@app.route('/')** : 기본 경로 (/)로 접속했을 때 이 함수를 호출  
- **@app.route('/folder/<folder_id>')**: /folder/<folder_id>로 접속할 때, 폴더 ID에 따라 하위 폴더를 표시  
- **def home(folder_id=None):** : folder_id=None: 폴더 ID가 없으면 기본 폴더에서 시작  
- **files = get_drive_files(folder_id)** : 지정한 폴더 ID에 있는 파일 목록을 가져  
- **return render_template('index.html', files=files)** : index.html 템플릿을 렌더링하면서 files 데이터를 전달하여 웹 페이지에 표시

```python
# Flask 애플리케이션 실행M
if __name__ == '__main__':
    app.run(debug=True)
```

- **if name == 'main':** : 현재 파일이 메인 프로그램으로 실행될 때만 Flask 앱을 실행  
- **app.run(debug=True)** : Flask 애플리케이션을 실행, debug=True는 개발 모드로 실행하여 코드가 수정되면 서버가 자동으로 다시 시작되고, 오류 발생 시 디버깅 정보를 표시함


