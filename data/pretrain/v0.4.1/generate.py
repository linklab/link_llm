"""Small synthetic document benchmark authored with AI assistance for this repository.

Topic-specific template families are held out together. The same seed regenerates
the exact JSONL bytes. The old pretrain/train.txt and valid.txt are never modified.
"""
import argparse
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v041_corpus', HERE/'corpus.py')
corpus = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corpus)

# A separate paragraph template per topic; family IDs reflect these actual templates.
# Names/times vary within a family and must never be split across partitions.
TEMPLATES = {
    'garden': (
        '{who}의 텃밭 수첩\n\n{when}, 화분 옆에 작은 잎이 보였다. {who}는 먼저 흙을 만져 보고 물이 필요한지 살폈다. 어제 그려 둔 잎과 오늘의 잎을 비교하니 모양이 조금 달랐다.',
        '물은 흙의 상태를 확인한 뒤 주었다. 잎에 생긴 작은 점은 따로 그려 두었다. 관찰한 모습만으로 원인을 단정하지 않고 며칠 더 지켜보기로 했다.',
        '다음 관찰에서는 화분의 위치도 기록했다. 창가와 안쪽에서 빛을 받는 시간이 달라 보였기 때문이다. {who}는 물을 준 날과 잎을 그린 날을 같은 표에 적었다.'),
    'library': (
        '도서관에서 찾은 질문\n\n책을 고르던 {who}에게 궁금한 문장이 생겼다. {when}에 읽은 이야기 속 주인공은 왜 돌아왔을까? 목차와 앞부분을 다시 펼쳐 단서를 찾았다.',
        '답을 바로 정하지 않고 관련된 문장 옆에 종이쪽지를 끼웠다. 책을 덮은 뒤에는 기억나는 내용을 자기 말로 썼다. 다시 읽으니 기억과 다른 부분도 있었다.',
        '{who}는 친구에게 자신의 해석을 들려주었다. 친구가 다른 대목을 가리키자 두 사람은 앞뒤 문장을 함께 읽었다. 같은 책에서도 서로 다른 질문이 나올 수 있었다.'),
    'cooking': (
        '함께 준비하는 한 끼\n\n{who}는 {when}부터 식사 준비를 맡았다. 쓸 재료와 그릇을 먼저 나누어 놓으니 빠진 물건을 찾기 쉬웠다. 준비 순서를 종이에 적고 할 일을 하나씩 지웠다.',
        '조리 중에는 사용한 도구를 정리하며 다음 단계를 기다렸다. 식사를 마친 사람들은 좋았던 점과 바꾸고 싶은 점을 이야기했다. 맛에 대한 의견은 서로 같지 않았다.',
        '남은 재료를 확인한 {who}는 다음 장보기 목록을 고쳤다. 이번에는 양이 많았던 재료를 줄이기로 했다. 다음 식사에서도 같은 기록을 참고할 생각이었다.'),
    'weather': (
        '창밖을 보는 시간\n\n{when}의 하늘은 어제와 달랐다. {who}는 창문 앞에 앉아 구름의 모양과 나뭇가지의 움직임을 그렸다. 그림 아래에는 관찰한 시간을 적었다.',
        '잠시 뒤 같은 자리에서 다시 보니 구름이 옮겨 가 있었다. 바람이 강해졌다고 느꼈지만 정확한 수치는 알 수 없었다. 기록에는 느낌과 직접 본 사실을 구분했다.',
        '여러 날의 그림을 모은 뒤 {who}는 비슷한 모습을 찾아 묶었다. 비가 온 날만 골라 보기도 했다. 짧은 관찰만으로 다음 날씨를 확신하지는 않았다.'),
    'music': (
        '한 마디씩 맞추기\n\n합주를 앞둔 {who}는 어려운 마디에 표시했다. {when}에는 곡 전체를 빠르게 연주하는 대신 짧은 부분만 느리게 반복했다. 손이 익숙해지면 다음 마디와 연결했다.',
        '혼자 연주할 때와 친구의 소리를 들을 때는 느낌이 달랐다. 서로 시작하는 순간을 맞추자 멜로디가 이어졌다. 틀린 부분에서는 멈추고 박자를 다시 세었다.',
        '연습을 마치며 {who}는 다음에 맞출 구간을 골랐다. 잘된 부분도 표시해 두니 연습 과정이 보였다. 완벽하게 끝내는 것보다 함께 들으며 고치는 시간이 중요했다.'),
    'woodwork': (
        '작은 선반의 설계도\n\n나무판 앞에서 {who}는 바로 작업을 시작하지 않았다. {when}에 종이 위로 만들 모양을 옮겼다. 물건을 올려둘 넓이와 놓을 자리를 함께 생각했다.',
        '치수를 확인한 다음 부품을 맞춰 보았다. 한쪽이 어긋나면 앞에서 적은 숫자부터 다시 읽었다. 겉모양만 보는 대신 받침이 닿는 곳도 살폈다.',
        '완성한 선반을 빈 상태로 놓고 흔들림을 확인했다. {who}는 처음 도면과 달라진 부분을 다른 색으로 표시했다. 다음 작업에서는 그 이유부터 떠올릴 수 있을 것이다.'),
    'river': (
        '다리 위의 관찰자\n\n{who}가 개울에 도착한 때는 {when}이었다. 물가에 들어가지 않고 다리에서 흐름을 바라보았다. 돌 주변에서는 물이 다른 방향으로 움직이는 것처럼 보였다.',
        '수첩 한쪽에는 직접 본 모습을 그리고 다른 쪽에는 궁금한 점을 적었다. 멀리 있는 작은 물체의 정체는 확인할 수 없었다. 모르는 부분은 빈칸으로 남겼다.',
        '돌아오는 길에 {who}는 관찰 위치를 지도에 표시했다. 같은 장소를 다시 찾으면 오늘의 기록과 비교할 수 있다. 물의 움직임은 날마다 같을 것이라고 가정하지 않았다.'),
    'market': (
        '목록을 들고 시장으로\n\n살 것이 많아 보여도 모두 필요한 것은 아니었다. {who}는 {when}에 장바구니를 들기 전에 집에 남은 물건부터 확인했다. 꼭 살 것과 나중에 살 것을 나눴다.',
        '시장에서는 여러 가게의 표시를 읽었다. 가격뿐 아니라 양이 얼마나 되는지도 보았다. 목록에 없던 물건은 바로 담지 않고 한 번 더 생각했다.',
        '{who}는 집에 돌아와 영수증과 목록을 나란히 놓았다. 예상한 금액과 실제 쓴 금액의 차이를 적었다. 남은 예산은 다음 방문을 위해 따로 표시해 두었다.'),
    'museum': (
        '그림 앞에서 한 걸음 뒤로\n\n전시실에 들어선 {who}의 눈에 커다란 그림이 들어왔다. {when}에는 사람이 적어 작품을 천천히 볼 수 있었다. 가까이에서는 색의 흔적을, 멀리에서는 전체 모양을 살폈다.',
        '처음 떠올린 생각을 적은 뒤 설명을 읽었다. 설명을 알고 다시 보니 지나쳤던 부분이 눈에 띄었다. 그래도 자신의 느낌을 모두 지우지는 않았다.',
        '전시실을 나온 {who}는 가장 기억나는 장면을 친구에게 말했다. 친구는 다른 작품을 골랐다. 두 사람은 각자 무엇을 보았는지 이야기하며 안내도를 다시 펼쳤다.'),
    'running': (
        '오늘의 속도 찾기\n\n{when}, 운동화를 신은 {who}는 먼저 천천히 걸었다. 몸의 느낌을 살피며 달릴 준비를 했다. 어제 빨리 달렸다고 오늘도 같은 속도를 내야 하는 것은 아니었다.',
        '길을 따라 움직이다 힘들어지면 속도를 낮췄다. 주변 사람과 간격을 두고 안전하게 지나갔다. 끝까지 빠르게 가는 것보다 무리하지 않는 것이 오늘의 목표였다.',
        '쉬는 동안 {who}는 거리보다 몸의 느낌을 먼저 적었다. 편했던 구간과 힘들었던 구간을 구분했다. 다음번에는 그 기록을 보고 출발 속도를 정하기로 했다.'),
    'sewing': (
        '천 위에 남은 작은 선\n\n{who}는 만들 주머니의 모양을 종이에 그려 두었다. {when}에 천을 펼쳐 선을 옮겼다. 접히는 방향을 확인하지 않으면 완성한 모양이 달라질 수 있었다.',
        '한 땀씩 이어 가다가 선이 어긋난 곳에서 멈췄다. 서둘러 덮기보다 앞부분을 다시 살폈다. 바느질한 뒤에는 천을 뒤집어 빠진 곳이 없는지 확인했다.',
        '주머니에 작은 물건을 넣어 본 {who}는 입구의 넓이를 점검했다. 다음 작품에는 손잡이를 달고 싶어졌다. 처음 그린 도안 옆에 새 생각을 그려 두었다.'),
    'astronomy': (
        '지도와 밤하늘 사이\n\n{who}는 {when}에 별자리 지도를 펼쳤다. 관찰할 수 있는 시간과 하늘의 상태부터 기록했다. 지도에 있는 별이 모두 눈에 보이는 것은 아니었다.',
        '불빛과 구름 때문에 구분하기 어려운 곳도 있었다. 확실히 본 점과 추측한 점을 다른 표시로 남겼다. 방향을 다시 맞추니 처음 찾던 모양과 비슷한 배열이 보였다.',
        '관찰을 마친 {who}는 다음에 확인할 위치를 골랐다. 보이지 않은 별도 조건과 함께 기록했다. 다른 날의 관찰과 비교할 때 필요한 정보이기 때문이다.'),
}
NAMES = ('민수', '지우', '서연', '하준', '수빈', '도윤', '예린', '준서')
TIMES = ('이른 아침', '점심 무렵', '늦은 오후', '해 질 무렵', '저녁 시간', '주말 오전')


def records(per_topic=40):
    if not 1 <= per_topic <= len(NAMES)*len(TIMES):
        raise ValueError('per_topic must be between 1 and 48 distinct combinations')
    result = []
    for topic, template in TEMPLATES.items():
        for i in range(per_topic):
            who, when = NAMES[i % len(NAMES)], TIMES[i // len(NAMES)]
            paragraphs = [part.format(who=who, when=when) for part in template[:2]]
            if i % 4 == 0:
                paragraphs.append(template[2].format(who=who, when=when))
            result.append({'id': f'{topic}-{i:03d}', 'text': '\n\n'.join(paragraphs),
                           'source': 'authored-v041', 'topic': topic,
                           'template_id': f'{topic}-journal-v1'})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--per-topic', type=int, default=40)
    args = parser.parse_args()
    sources = {'authored-v041': {'uri': 'repo:data/pretrain/v0.4.1/generate.py',
               'license': 'MIT (repository LICENSE)',
               'usage': 'Synthetic educational examples; retain repository license notice when redistributing.',
               'description': 'AI-assisted templates authored for this version; no web scraping or external corpus.'}}
    manifest = corpus.prepare(records(args.per_topic), sources, args.output_dir, args.seed,
                              provenance={'generator_sha256': corpus.sha256(Path(__file__).read_bytes()),
                              'pipeline_sha256': corpus.sha256((HERE/'corpus.py').read_bytes()),
                              'license_sha256': corpus.sha256((HERE.parents[2]/'LICENSE').read_bytes()),
                              'per_topic': args.per_topic})
    print({name: info['documents'] for name, info in manifest['splits'].items()})


if __name__ == '__main__':
    main()
