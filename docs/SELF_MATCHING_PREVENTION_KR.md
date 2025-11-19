# 자체 매칭 방지 기능 구현

## 개요

자체 매칭 방지 기능을 추가하여 자신의 오픈 주문과 클로즈 주문이 서로 매칭되는 것을 방지합니다. 이를 통해 불필요한 수수료를 절감하면서도 볼륨 거래를 유지할 수 있습니다.

## 구현 내용

### 1. 자체 매칭 검사 함수 (`_check_self_matching`)

새로운 주문을 배치하기 전에 반대편 활성 주문과 겹치는지 확인하는 함수를 추가했습니다.

**기능**:
- 모든 활성 주문 조회
- 반대편 주문 필터링 (buy 주문이면 sell 주문 확인, 그 반대)
- 가격 겹침 확인:
  - Buy 주문: 새 buy 가격 >= 기존 sell 가격이면 매칭
  - Sell 주문: 새 sell 가격 <= 기존 buy 가격이면 매칭

**위치**: `trading_bot.py`의 `_check_self_matching` 메서드

### 2. 클로즈 주문 배치 시 자체 매칭 방지

클로즈 주문을 배치하기 전에 자체 매칭을 확인하고, 겹칠 경우 가격을 자동 조정합니다.

**동작 방식**:
1. 클로즈 주문 가격 계산 (기존 로직)
2. 자체 매칭 검사 실행
3. 매칭이 감지되면:
   - **Sell 주문**: best_bid 기준으로 가격 상향 조정 (0.1% 버퍼 추가)
   - **Buy 주문**: best_ask 기준으로 가격 하향 조정 (0.1% 버퍼 감소)
4. 조정된 가격으로 주문 배치

**적용 위치**:
- 완전 체결 시 클로즈 주문 배치 (`_handle_order_result`)
- 부분 체결 후 클로즈 주문 배치 (`_handle_order_result`)

### 3. 설정 옵션

`TradingConfig`에 `prevent_self_matching` 플래그를 추가했습니다.

- **기본값**: `True` (기본적으로 활성화)
- **비활성화**: 설정을 `False`로 변경하면 자체 매칭 방지 기능이 작동하지 않습니다

## 사용 방법

기본적으로 자체 매칭 방지 기능이 활성화되어 있습니다. 별도의 설정 없이 자동으로 작동합니다.

```python
# 기본 사용 (자체 매칭 방지 활성화)
config = TradingConfig(
    ticker="ETH",
    # ... 기타 설정
    prevent_self_matching=True  # 기본값
)

# 자체 매칭 방지 비활성화 (필요한 경우)
config = TradingConfig(
    ticker="ETH",
    # ... 기타 설정
    prevent_self_matching=False
)
```

## 예상 효과

### 수수료 절감
- **자체 매칭 방지**: 양쪽 주문 모두 수수료가 발생하는 것을 방지
- **예상 절감**: 월 거래량의 10-20% 수수료 절감 가능

### 볼륨 유지
- 자체 매칭을 방지하지만, 가격을 조정하여 주문을 배치하므로 볼륨은 유지됩니다
- 조정된 가격은 시장 상황에 맞게 최적화됩니다

### 로그 모니터링
자체 매칭이 감지되고 가격이 조정될 때마다 로그가 기록됩니다:

```
[SELF-MATCH PREVENTION] New sell order @ 2000.50 would match existing buy order @ 2000.40
[SELF-MATCH PREVENTION] Adjusted close sell price from 2000.50 to 2002.00
```

## 기술적 세부사항

### 가격 조정 로직

**Sell 주문 조정**:
```python
adjusted_price = best_bid * (1 + take_profit/100 + 0.001)
close_price = max(original_close_price, adjusted_price)
```

**Buy 주문 조정**:
```python
adjusted_price = best_ask * (1 - take_profit/100 - 0.001)
close_price = min(original_close_price, adjusted_price)
```

### 에러 처리

- 자체 매칭 검사 중 오류 발생 시, 주문은 정상적으로 진행됩니다 (fail-safe)
- 로그에 오류가 기록되지만 거래는 중단되지 않습니다

## 향후 개선 사항

1. **동적 버퍼 조정**: 고정된 0.1% 버퍼 대신 시장 변동성에 따라 동적 조정
2. **주문북 깊이 분석**: 더 정교한 가격 조정을 위한 주문북 깊이 분석
3. **통계 추적**: 자체 매칭 방지로 인한 수수료 절감량 추적 및 리포트

## 주의사항

1. **가격 조정**: 자체 매칭을 방지하기 위해 가격이 조정될 수 있으며, 이는 take-profit 목표와 약간 다를 수 있습니다
2. **성능**: 활성 주문이 많을 경우 검사 시간이 약간 증가할 수 있습니다
3. **거래소 호환성**: 모든 거래소에서 동일하게 작동하지만, 일부 거래소에서는 추가 최적화가 필요할 수 있습니다
