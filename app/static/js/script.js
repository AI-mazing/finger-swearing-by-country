$(function () {
    // #btnArea 내부의 버튼 클릭 이벤트 핸들러
    $("div#btnArea>button").click(function () {
        // 버튼 클릭 시 /global 페이지로 이동
        location.href = "/global";
    });

    // #countryForm 내부의 라벨 클릭 이벤트 핸들러
    $("form#countryForm label").click(function () {
        // 클릭한 라벨 내부의 input 요소에서 값을 가져와서
        let selectCountry = $(this).find("input").val();
        // 로컬 스토리지에 선택된 값을 저장
        localStorage.setItem('selectedRadioValue', selectCountry);
        // /situation1 페이지로 이동하면서 선택된 국가 값을 쿼리 파라미터로 전달
        location.href = "/situation1?country=" + selectCountry;
    });

    // #next 내부의 span 클릭 이벤트 핸들러
    $("div#next span").click(function () {
        // 로컬 스토리지에서 저장된 값을 가져와서
        let selectVal = localStorage.getItem('selectedRadioValue');
        // /situation2 페이지로 이동하면서 선택된 국가 값을 쿼리 파라미터로 전달
        location.href = "/situation2?country=" + selectVal;
        // 선택된 국가 값을 콘솔에 출력
        console.log("selectCountry : " + selectVal);
    });

    // 페이지 로드 완료 후 실행되는 함수
    window.onload = function () {
        // 페이지의 hidden input 요소에서 session_id 값을 가져옴
        const sessionId = document.getElementById('session_id').value;
        if (sessionId) {
            // WebSocket URL을 생성
            const wsUrl = `ws://${window.location.host}/ws/${sessionId}`;
            console.log(`Connecting to WebSocket at ${wsUrl}`);
            
            // WebSocket 객체 생성 및 연결
            const ws = new WebSocket(wsUrl);

            // WebSocket 연결이 열릴 때 호출되는 함수
            ws.onopen = function () {
                console.log("WebSocket connection established.");
            };

            // WebSocket으로부터 메시지를 수신할 때 호출되는 함수
            ws.onmessage = function (event) {
                // 수신한 데이터가 Blob 객체일 때 처리
                if (event.data instanceof Blob) {
                    // Blob 데이터를 URL로 변환하여 비디오 소스에 설정
                    const urlObject = URL.createObjectURL(event.data);
                    document.getElementById('videoFeed').src = urlObject;
                    // 100ms 후에 URL 객체를 해제
                    setTimeout(() => URL.revokeObjectURL(urlObject), 100);
                } else if (event.data === "gesture_recognized") {
                    // 제스처 인식이 완료된 경우 /success 페이지로 요청
                    fetch('/success', {
                        method: 'GET',
                        credentials: 'include'  // 쿠키 포함
                    }).then(response => {
                        // 응답이 리디렉션인 경우 해당 URL로 이동
                        if (response.redirected) {
                            window.location.href = response.url;
                        } else {
                            window.location.href = '/success';
                        }
                    });
                } else {
                    // 기타 메시지는 콘솔에 출력
                    console.log("Received message:", event.data);
                }
            };

            // WebSocket 에러 발생 시 호출되는 함수
            ws.onerror = function (error) {
                console.error("WebSocket error:", error);
            };

            // WebSocket 연결이 닫힐 때 호출되는 함수
            ws.onclose = function () {
                console.log("WebSocket connection closed.");
            };
        } else {
            // sessionId가 없는 경우 에러 메시지 출력
            console.error("No sessionId found.");
        }
    };
});
