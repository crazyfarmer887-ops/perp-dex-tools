# 수수료 추적 기능 구현 완료

## 구현된 기능

### 1. 수수료 추적 시스템 (`helpers/fee_tracker.py`)

**주요 기능**:
- 각 거래의 수수료 자동 계산 (Maker/Taker 구분)
- 거래소별 수수료 구조 지원
- CSV 파일로 수수료 로그 저장
- 실시간 수수료 통계 제공

**추적 항목**:
- 총 거래량 (Total Volume)
- 총 수수료 (Total Fees)
- Maker/Taker 수수료 분리
- OPEN/CLOSE 주문별 수수료
- 거래당 평균 수수료
- 수수료 비율

### 2. 트레이딩 봇 통합

**변경 사항**:
- 모든 체결된 주문에 대해 수수료 자동 추적
- Boost 모드에서 Taker 수수료 자동 인식
- 주기적 로그에 수수료 정보 표시
- 봇 종료 시 수수료 요약 출력

## 사용 방법

### 자동 추적
수수료 추적은 자동으로 작동합니다. 봇을 실행하면:
1. 모든 체결된 주문의 수수료가 자동으로 계산됩니다
2. `logs/{exchange}_{ticker}_fees.csv` 파일에 수수료 로그가 저장됩니다
3. 60초마다 로그에 현재까지의 수수료 정보가 표시됩니다
4. 봇 종료 시 전체 수수료 요약이 출력됩니다

### 수수료 로그 파일 형식
CSV 파일에는 다음 정보가 저장됩니다:
- Timestamp: 거래 시간
- OrderID: 주문 ID
- OrderType: OPEN 또는 CLOSE
- Side: buy 또는 sell
- Quantity: 거래 수량
- Price: 거래 가격
- Notional: 거래 금액 (Quantity × Price)
- FeeType: MAKER 또는 TAKER
- FeeRate: 수수료 비율 (%)
- FeeAmount: 수수료 금액
- Status: 주문 상태

### 수수료 요약 예시
```
============================================================
Fee Summary - EDGEX ETH
============================================================
Total Volume: $10,000.00
Total Fees: $4.0000
  - Maker Fees: $4.0000
  - Taker Fees: $0.0000
  - Open Order Fees: $2.0000
  - Close Order Fees: $2.0000
Transaction Count: 100
Average Fee per Transaction: $0.0400
Fee Percentage: 0.0400%
Maker Fee Rate: 0.02%
Taker Fee Rate: 0.05%
============================================================
```

## 거래소별 수수료 설정

현재 기본 수수료 구조는 `fee_tracker.py`의 `EXCHANGE_FEES` 딕셔너리에 정의되어 있습니다:

```python
EXCHANGE_FEES = {
    'edgex': {'maker': Decimal('0.02'), 'taker': Decimal('0.05')},
    'backpack': {'maker': Decimal('0.02'), 'taker': Decimal('0.04')},
    'extended': {'maker': Decimal('0.02'), 'taker': Decimal('0.05')},
    # ... 기타 거래소
}
```

**VIP 레벨이나 리베이트가 있는 경우**:
`helpers/fee_tracker.py` 파일의 `EXCHANGE_FEES` 딕셔너리를 수정하여 실제 수수료 구조를 반영할 수 있습니다.

## 다음 단계 제안

### 즉시 구현 가능한 기능

1. **수수료 알림 시스템**
   - 일일 수수료가 특정 임계값을 초과하면 알림
   - Taker 수수료가 예상보다 높을 때 경고

2. **수수료 리포트 생성**
   - 일일/주간/월간 수수료 리포트 자동 생성
   - 수수료 추세 분석 그래프

3. **Maker 보장 강화**
   - 주문 가격을 더 보수적으로 설정하여 maker 확률 향상
   - 주문이 즉시 체결되지 않도록 가격 조정

### 중기 구현 가능한 기능

4. **주문 집계 기능**
   - 비슷한 가격의 여러 close 주문을 하나로 통합
   - 주문 관리 효율화

5. **수수료 인식 그리드 전략**
   - 수수료를 고려한 최소 그리드 간격 계산
   - 수수료를 커버할 수 있는 최소 take-profit 설정

## 참고사항

- 수수료 추적은 체결된 주문에 대해서만 수행됩니다
- 취소된 주문이나 부분 체결 후 취소된 주문의 부분 체결분도 추적됩니다
- Boost 모드에서는 CLOSE 주문이 Taker로 인식됩니다
- 실제 거래소의 수수료 구조가 다를 수 있으므로, 필요시 `EXCHANGE_FEES`를 업데이트하세요

## 문제 해결

**수수료가 예상과 다르게 계산되는 경우**:
1. `helpers/fee_tracker.py`의 `EXCHANGE_FEES` 확인
2. 실제 거래소의 수수료 구조 확인
3. VIP 레벨이나 리베이트 프로그램 확인

**수수료 로그 파일이 생성되지 않는 경우**:
1. `logs` 디렉토리 권한 확인
2. 디스크 공간 확인
3. 로그 파일 경로 확인
