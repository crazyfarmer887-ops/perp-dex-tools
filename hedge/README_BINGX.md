# BingX Hedge Position Mode

빙엑스(BingX) 거래소에서 양방향 헷지 포지션을 자동으로 관리하는 트레이딩 봇입니다.

## 주요 기능

1. **양방향 시장가 진입**: 현재 시장 가격에서 롱/숏 포지션을 동시에 진입
2. **평균가 기준 TP/SL**: 두 포지션의 평균 진입가를 기준으로 TP/SL 설정
3. **자동 리밋 주문**: TP/SL은 리밋 주문으로 자동 배치

## 동작 원리

### 1. 포지션 진입
- 지정된 수량으로 롱 포지션 (시장가 매수)
- 동일한 수량으로 숏 포지션 (시장가 매도)
- 두 주문은 동시에 실행됨

### 2. 평균가 계산
예시: 
- 롱 포지션: 90,000 USDT에 체결
- 숏 포지션: 100,000 USDT에 체결
- 평균 진입가: (90,000 + 100,000) / 2 = 95,000 USDT

### 3. TP/SL 설정 (ROI 10% 예시)
평균가 95,000 기준으로:

**롱 포지션:**
- TP (Take Profit): 95,000 × 1.1 = 104,500 USDT (매도 리밋)
- SL (Stop Loss): 95,000 × 0.9 = 85,500 USDT (매도 리밋)

**숏 포지션:**
- TP (Take Profit): 95,000 × 0.9 = 85,500 USDT (매수 리밋)
- SL (Stop Loss): 95,000 × 1.1 = 104,500 USDT (매수 리밋)

## 사용 방법

### 환경 설정

`.env` 파일에 BingX API 자격 증명을 설정하세요:

```bash
BINGX_API_KEY=your_bingx_api_key
BINGX_API_SECRET=your_bingx_api_secret
BINGX_ENVIRONMENT=prod  # 또는 testnet
```

### 실행 명령

#### 기본 실행 (1회)
```bash
python3 hedge/hedge_mode_bingx.py BTC 0.001 --tp-roi 10 --sl-roi 10
```

#### 파라미터 설명
- `ticker`: 거래 페어 (예: BTC, ETH)
- `quantity`: 각 포지션 크기
- `--tp-roi`: 이익 실현 ROI % (선택사항)
- `--sl-roi`: 손절 ROI % (선택사항)
- `--iterations`: 반복 횟수 (기본값: 1)
- `--sleep`: 반복 간 대기 시간(초) (기본값: 0)

### 실행 예시

#### 1. BTC 0.001개, TP/SL 5%
```bash
python3 hedge/hedge_mode_bingx.py BTC 0.001 --tp-roi 5 --sl-roi 5
```

#### 2. ETH 0.1개, TP 10% / SL 5%
```bash
python3 hedge/hedge_mode_bingx.py ETH 0.1 --tp-roi 10 --sl-roi 5
```

#### 3. 3회 반복, 각 사이클 60초 대기
```bash
python3 hedge/hedge_mode_bingx.py BTC 0.001 --tp-roi 10 --sl-roi 10 --iterations 3 --sleep 60
```

## 로그 파일

실행 로그는 `logs/` 디렉토리에 저장됩니다:
- `logs/bingx_{ticker}_hedge_log.txt`: 상세 실행 로그

## 주의 사항

1. **리스크 관리**: 양방향 포지션은 변동성이 큰 시장에서 손실을 볼 수 있습니다
2. **수수료**: 시장가 주문과 리밋 주문의 수수료를 고려하세요
3. **최소 주문량**: BingX의 최소 주문 수량 제한을 확인하세요
4. **API 제한**: API 레이트 리밋을 초과하지 않도록 주의하세요

## 종료 방법

프로그램 실행 중 `Ctrl+C`를 누르면 안전하게 종료됩니다.
- 남은 포지션이 있다면 자동으로 청산을 시도합니다
- 로그 파일에 종료 기록이 남습니다

## 트러블슈팅

### API 오류
- API 키와 시크릿이 올바른지 확인
- API 권한에 선물 거래가 허용되어 있는지 확인

### 주문 실패
- 계정 잔액이 충분한지 확인
- 레버리지 설정이 적절한지 확인
- 최소 주문 수량을 만족하는지 확인

### TP/SL 미체결
- 시장 가격이 TP/SL 레벨에 도달하지 않으면 타임아웃됩니다
- 타임아웃 시 남은 포지션은 자동 청산됩니다