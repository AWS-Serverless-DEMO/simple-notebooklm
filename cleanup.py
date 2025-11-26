#!/usr/bin/env python3
"""
AWS S3 Vectors Cleanup Script

This script helps clean up resources after project presentation.
Use this to delete vectors and free up AWS resources to avoid unnecessary costs.
"""

import argparse
import sys
from typing import Optional
from config import Config
from utils import S3VectorStore


def confirm_action(message: str) -> bool:
    """Ask user for confirmation before proceeding"""
    response = input(f"\n{message} (yes/no): ").strip().lower()
    return response in ['yes', 'y']


def list_all_documents(vector_store: S3VectorStore) -> None:
    """List all documents in the vector store"""
    print("\n" + "="*60)
    print("📋 저장된 문서 목록")
    print("="*60)

    try:
        documents = vector_store.list_documents()

        if not documents:
            print("\n저장된 문서가 없습니다.")
            return

        for i, doc in enumerate(documents, 1):
            print(f"\n{i}. 문서: {doc['document']}")
            print(f"   타입: {doc['source_type']}")
            print(f"   청크 수: {doc['chunk_count']}")
            print(f"   페이지 수: {doc['page_count']}")
            if doc['pages']:
                print(f"   페이지: {doc['pages'][:5]}" + ("..." if len(doc['pages']) > 5 else ""))

        print(f"\n총 {len(documents)}개 문서, {sum(d['chunk_count'] for d in documents)}개 벡터")

    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        sys.exit(1)


def delete_document(vector_store: S3VectorStore, document_name: str) -> None:
    """Delete a specific document"""
    print(f"\n🗑️  문서 삭제 중: {document_name}")

    try:
        result = vector_store.delete_vectors_by_document(document_name)

        if result['deleted_count'] == 0:
            print(f"⚠️  {result['message']}")
        else:
            print(f"✅ {result['message']}")

    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        sys.exit(1)


def delete_all_vectors(vector_store: S3VectorStore) -> None:
    """Delete all vectors in the index"""
    print("\n🗑️  모든 벡터 삭제 중...")

    try:
        result = vector_store.delete_all_vectors()

        if result['deleted_count'] == 0:
            print(f"⚠️  {result['message']}")
        else:
            print(f"✅ {result['message']}")

    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        sys.exit(1)


def delete_index(vector_store: S3VectorStore) -> None:
    """Delete the vector index"""
    print("\n🗑️  Vector Index 삭제 중...")

    try:
        result = vector_store.delete_index()

        if result['status'] == 'deleted':
            print(f"✅ Index 삭제 완료: {result['index_name']}")
        elif result['status'] == 'not_found':
            print(f"⚠️  Index를 찾을 수 없습니다: {result['index_name']}")

    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        sys.exit(1)


def delete_bucket(vector_store: S3VectorStore) -> None:
    """Delete the vector bucket"""
    print("\n🗑️  Vector Bucket 삭제 중...")

    try:
        result = vector_store.delete_bucket()

        if result['status'] == 'deleted':
            print(f"✅ Bucket 삭제 완료: {result['bucket_name']}")
        elif result['status'] == 'not_found':
            print(f"⚠️  Bucket을 찾을 수 없습니다: {result['bucket_name']}")

    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        print("\n💡 Bucket 삭제 전 모든 Index를 먼저 삭제해야 합니다.")
        sys.exit(1)


def delete_everything(vector_store: S3VectorStore) -> None:
    """Delete all vectors, index, and bucket"""
    print("\n🗑️  모든 리소스 삭제 중 (벡터 → 인덱스 → 버킷)...")

    # Step 1: Delete all vectors
    print("\n[1/3] 벡터 삭제 중...")
    try:
        result = vector_store.delete_all_vectors()
        if result['deleted_count'] > 0:
            print(f"✅ {result['deleted_count']}개 벡터 삭제 완료")
        else:
            print("⚠️  삭제할 벡터 없음")
    except Exception as e:
        print(f"⚠️  벡터 삭제 실패 (계속 진행): {str(e)}")

    # Step 2: Delete index
    print("\n[2/3] 인덱스 삭제 중...")
    try:
        result = vector_store.delete_index()
        if result['status'] == 'deleted':
            print(f"✅ Index 삭제 완료")
        else:
            print(f"⚠️  Index를 찾을 수 없음")
    except Exception as e:
        print(f"❌ Index 삭제 실패: {str(e)}")
        print("⚠️  Bucket 삭제를 건너뜁니다.")
        return

    # Step 3: Delete bucket
    print("\n[3/3] 버킷 삭제 중...")
    try:
        result = vector_store.delete_bucket()
        if result['status'] == 'deleted':
            print(f"✅ Bucket 삭제 완료")
        else:
            print(f"⚠️  Bucket을 찾을 수 없음")
    except Exception as e:
        print(f"❌ Bucket 삭제 실패: {str(e)}")

    print("\n" + "="*60)
    print("✅ 모든 리소스 삭제 완료!")
    print("="*60)


def interactive_mode(vector_store: S3VectorStore) -> None:
    """Interactive cleanup mode with menu"""
    while True:
        print("\n" + "="*60)
        print("🧹 Simple NotebookLM - 리소스 정리")
        print("="*60)
        print("\n1. 저장된 문서 목록 보기")
        print("2. 특정 문서 삭제")
        print("3. 모든 벡터 삭제 (전체 초기화)")
        print("4. 전체 리소스 삭제 (벡터 + 인덱스 + 버킷)")
        print("5. 종료")

        choice = input("\n선택하세요 (1-5): ").strip()

        if choice == '1':
            list_all_documents(vector_store)

        elif choice == '2':
            list_all_documents(vector_store)
            document_name = input("\n삭제할 문서 이름을 입력하세요: ").strip()

            if document_name:
                if confirm_action(f"'{document_name}' 문서의 모든 벡터를 삭제하시겠습니까?"):
                    delete_document(vector_store, document_name)
                else:
                    print("취소되었습니다.")
            else:
                print("문서 이름이 입력되지 않았습니다.")

        elif choice == '3':
            list_all_documents(vector_store)

            if confirm_action("⚠️  경고: 모든 벡터를 삭제하시겠습니까? 이 작업은 되돌릴 수 없습니다!"):
                if confirm_action("정말로 확실합니까? 한 번 더 확인합니다."):
                    delete_all_vectors(vector_store)
                else:
                    print("취소되었습니다.")
            else:
                print("취소되었습니다.")

        elif choice == '4':
            print("\n⚠️  위험: 모든 AWS 리소스(벡터, 인덱스, 버킷)를 삭제합니다!")
            print("이 작업은 되돌릴 수 없으며, 프로젝트를 다시 시작하려면 리소스를 재생성해야 합니다.")

            if confirm_action("⚠️  정말로 모든 리소스를 삭제하시겠습니까?"):
                if confirm_action("⚠️⚠️  마지막 확인: 벡터, 인덱스, 버킷을 모두 삭제합니까?"):
                    delete_everything(vector_store)
                else:
                    print("취소되었습니다.")
            else:
                print("취소되었습니다.")

        elif choice == '5':
            print("\n종료합니다.")
            break

        else:
            print("\n잘못된 선택입니다. 1-5 사이의 숫자를 입력하세요.")


def main():
    """Main cleanup function"""
    parser = argparse.ArgumentParser(
        description="AWS S3 Vectors cleanup script for Simple NotebookLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 인터랙티브 모드 (권장)
  python cleanup.py

  # 저장된 문서 목록만 보기
  python cleanup.py --list

  # 특정 문서 삭제 (확인 없이)
  python cleanup.py --delete "document.pdf" --force

  # 모든 벡터 삭제 (확인 없이)
  python cleanup.py --delete-all --force

  # 전체 리소스 삭제 (벡터 + 인덱스 + 버킷, 확인 없이)
  python cleanup.py --delete-everything --force
        """
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='저장된 문서 목록만 출력'
    )

    parser.add_argument(
        '--delete',
        type=str,
        metavar='DOCUMENT_NAME',
        help='특정 문서 삭제'
    )

    parser.add_argument(
        '--delete-all',
        action='store_true',
        help='모든 벡터 삭제'
    )

    parser.add_argument(
        '--delete-everything',
        action='store_true',
        help='전체 리소스 삭제 (벡터 + 인덱스 + 버킷)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='확인 없이 즉시 실행'
    )

    args = parser.parse_args()

    # Validate configuration
    print("⚙️  설정 확인 중...")
    try:
        Config.validate()
        print(f"✅ AWS 리전: {Config.AWS_REGION}")
        print(f"✅ Vector Bucket: {Config.S3_VECTOR_BUCKET_NAME}")
        print(f"✅ Vector Index: {Config.S3_VECTOR_INDEX_NAME}")
    except ValueError as e:
        print(f"\n❌ 설정 오류: {str(e)}")
        print("Please set up your `.env` file with required AWS credentials.")
        sys.exit(1)

    # Initialize vector store
    vector_store = S3VectorStore()

    # Execute based on arguments
    if args.list:
        list_all_documents(vector_store)

    elif args.delete:
        if args.force or confirm_action(f"'{args.delete}' 문서를 삭제하시겠습니까?"):
            delete_document(vector_store, args.delete)
        else:
            print("취소되었습니다.")

    elif args.delete_all:
        if args.force:
            delete_all_vectors(vector_store)
        else:
            list_all_documents(vector_store)
            if confirm_action("⚠️  모든 벡터를 삭제하시겠습니까?"):
                if confirm_action("정말로 확실합니까?"):
                    delete_all_vectors(vector_store)
                else:
                    print("취소되었습니다.")
            else:
                print("취소되었습니다.")

    elif args.delete_everything:
        if args.force:
            delete_everything(vector_store)
        else:
            print("\n⚠️  위험: 모든 AWS 리소스(벡터, 인덱스, 버킷)를 삭제합니다!")
            print("이 작업은 되돌릴 수 없으며, 프로젝트를 다시 시작하려면 리소스를 재생성해야 합니다.\n")
            if confirm_action("⚠️  정말로 모든 리소스를 삭제하시겠습니까?"):
                if confirm_action("⚠️⚠️  마지막 확인: 벡터, 인덱스, 버킷을 모두 삭제합니까?"):
                    delete_everything(vector_store)
                else:
                    print("취소되었습니다.")
            else:
                print("취소되었습니다.")

    else:
        # No arguments - run interactive mode
        interactive_mode(vector_store)


if __name__ == "__main__":
    main()
