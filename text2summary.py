def text2summary(input_text, client):

    # 시스템 역할과 응답 형식 지정
    system_role = '''당신은 119 구급 콜센터 직원입니다. 주어진 상황을 요약하세요.
    응답은 다음의 형식을 지켜주세요.
    \"증상 요약\"
    '''

    # 입력데이터를 GPT-3.5-turbo에 전달하고 답변 받아오기
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": system_role
            },
            {
                "role": "user",
                "content": input_text
            }
        ]
    )

    # 응답 받기
    text_summary = response.choices[0].message.content
    return text_summary
