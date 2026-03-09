import os
import pandas as pd
import win32com.client as win32

def extract_drm_excel_to_pickle(input_folder, output_folder):
    # 결과물을 저장할 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 1. 엑셀 애플리케이션의 독립된 새 인스턴스 백그라운드 실행
    excel = win32.DispatchEx('Excel.Application')
    
    # 2. 자동화 중단 방지를 위한 필수 방어 설정
    excel.Visible = False           # 엑셀 UI 숨김
    excel.DisplayAlerts = False     # 저장 확인, 경고창 등 모든 팝업 무시
    excel.ScreenUpdating = False    # 화면 렌더링 중지 (속도 향상)
    excel.EnableEvents = False      # 내부 이벤트 트리거 중지
    excel.AutomationSecurity = 3    # 매크로 자동 실행을 완전히 차단
    
    try:
        # 폴더 내 모든 파일 탐색
        for filename in os.listdir(input_folder):
            if filename.endswith(('.xlsx', '.xls')):
                # COM 객체는 반드시 절대 경로를 사용해야 함
                filepath = os.path.abspath(os.path.join(input_folder, filename))
                
                try:
                    # 3. 읽기 전용으로 파일 열기 (외부 링크 업데이트 무시)
                    wb = excel.Workbooks.Open(filepath, ReadOnly=True, UpdateLinks=0)
                    sheet = wb.Sheets(1) # 첫 번째 워크시트 선택
                    
                    # 4. 데이터 추출 (UsedRange.Value를 통해 한 번에 튜플 형태로 가져와 속도 극대화)
                    data = sheet.UsedRange.Value
                    
                    if data:
                        # 첫 번째 행을 헤더(컬럼명)로 지정하고 나머지를 데이터로 사용
                        df = pd.DataFrame(list(data[1:]), columns=data)
                        
                        # 5. 데이터프레임 정상 진입 확인 후 Pickle로 직렬화하여 저장
                        pkl_filename = os.path.splitext(filename) + '.pkl'
                        pkl_filepath = os.path.abspath(os.path.join(output_folder, pkl_filename))
                        
                        df.to_pickle(pkl_filepath)
                        print(f"[성공] 데이터프레임 로드 및 피클 저장 완료: {pkl_filename}")
                        
                except Exception as e:
                    print(f"[오류] {filename} 파일 처리 중 문제 발생: {e}")
                
                finally:
                    # 6. 메모리 누수를 막기 위해 개별 파일을 저장하지 않고 명시적으로 닫기
                    if 'wb' in locals():
                        wb.Close(SaveChanges=False)
                        
    finally:
        # 7. 엑셀 설정 원상 복구 및 프로세스 킬 (가비지 컬렉션 유도)
        excel.DisplayAlerts = True
        excel.ScreenUpdating = True
        excel.EnableEvents = True
        excel.Quit()
        print("모든 작업이 완료되어 엑셀 프로세스를 종료했습니다.")

# 사용 예시 (경로를 실제 환경에 맞게 수정하세요)
# input_directory = './drm_encrypted_files'
# output_directory = './decrypted_pickles'
# extract_drm_excel_to_pickle(input_directory, output_directory)
