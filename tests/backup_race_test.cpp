#include <gtest/gtest.h>
#include <filesystem>
#include <string>
#include <thread>
#include <chrono>
#include <unistd.h>

#include "storage/backup_store.hpp"
#include "core/ndd.hpp"

namespace fs = std::filesystem;

// ============================================================
// BackupStore Unit Tests
// Tests the guard state-machine that deleteIndex relies on.
// ============================================================

class BackupStoreTest : public ::testing::Test {
protected:
    std::string temp_dir;
    std::unique_ptr<BackupStore> store;

    void SetUp() override {
        temp_dir = "./test_backup_store_" + std::to_string(rand());
        fs::create_directories(temp_dir);
        store = std::make_unique<BackupStore>(temp_dir);
    }

    void TearDown() override {
        store.reset();
        if (fs::exists(temp_dir)) fs::remove_all(temp_dir);
    }
};

TEST_F(BackupStoreTest, NoActiveBackupByDefault) {
    EXPECT_FALSE(store->hasActiveBackup("alice"));
    EXPECT_FALSE(store->getActiveBackup("alice").has_value());
}

TEST_F(BackupStoreTest, SetAndGetActiveBackup) {
    store->setActiveBackup("alice", "alice/myindex", "backup1");

    EXPECT_TRUE(store->hasActiveBackup("alice"));

    auto active = store->getActiveBackup("alice");
    ASSERT_TRUE(active.has_value());
    EXPECT_EQ(active->index_id, "alice/myindex");
    EXPECT_EQ(active->backup_name, "backup1");
}

TEST_F(BackupStoreTest, ClearActiveBackup) {
    store->setActiveBackup("alice", "alice/myindex", "backup1");
    EXPECT_TRUE(store->hasActiveBackup("alice"));

    store->clearActiveBackup("alice");
    EXPECT_FALSE(store->hasActiveBackup("alice"));
    EXPECT_FALSE(store->getActiveBackup("alice").has_value());
}

TEST_F(BackupStoreTest, ActiveBackupIsolatedPerUser) {
    store->setActiveBackup("alice", "alice/myindex", "backup1");

    EXPECT_TRUE(store->hasActiveBackup("alice"));
    EXPECT_FALSE(store->hasActiveBackup("bob"));
}

TEST_F(BackupStoreTest, ActiveBackupMatchesCorrectIndex) {
    store->setActiveBackup("alice", "alice/index_a", "backup1");

    auto active = store->getActiveBackup("alice");
    ASSERT_TRUE(active.has_value());

    // Simulates the guard logic in deleteIndex:
    // a backup for index_a must NOT block deletion of index_b.
    EXPECT_EQ(active->index_id, "alice/index_a");
    EXPECT_NE(active->index_id, "alice/index_b");
}

// ============================================================
// IndexManager Integration Tests
// Verifies that deleteIndex throws when a backup is in progress
// for the same index, and succeeds when no backup is active.
// ============================================================

class DeleteIndexRaceTest : public ::testing::Test {
protected:
    std::string temp_dir;
    std::unique_ptr<IndexManager> manager;

    void SetUp() override {
        // Use pid + address of this to get a unique path even across crashed runs.
        temp_dir = "./test_delete_race_" + std::to_string(::getpid())
                   + "_" + std::to_string(reinterpret_cast<uintptr_t>(this));
        if (fs::exists(temp_dir)) fs::remove_all(temp_dir);
        fs::create_directories(temp_dir);
        manager = std::make_unique<IndexManager>(10, temp_dir);
    }

    void TearDown() override {
        // Poll until any detached backup thread clears the active flag, then
        // destroy the manager. Without this, the thread can access destroyed
        // members and crash with SIGSEGV.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < deadline) {
            // Cheap proxy: backup clears the flag when it finishes (success or error)
            bool any_active = false;
            for (const char* user : {"testuser", "alice", "bob"}) {
                if (manager->getActiveBackup(user).has_value()) { any_active = true; break; }
            }
            if (!any_active) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        manager.reset();
        if (fs::exists(temp_dir)) fs::remove_all(temp_dir);
    }

    static IndexConfig makeConfig() {
        return IndexConfig{
            .dim            = 4,
            .max_elements   = 100,
            .space_type_str = "cosine",
            .M              = 4,
            .ef_construction = 16,
            .quant_level    = ndd::quant::QuantizationLevel::INT8,
            .checksum       = 0,
        };
    }
};

// deleteIndex must throw std::runtime_error when a backup is in progress.
// createBackupAsync sets the active-backup flag before detaching the worker
// thread, so the flag is guaranteed to be set by the time the call returns.
TEST_F(DeleteIndexRaceTest, DeleteIndexThrowsWhenBackupActive) {
    const std::string index_id = "testuser/myindex";
    ASSERT_NO_THROW(manager->createIndex(index_id, makeConfig()));

    auto [ok, msg] = manager->createBackupAsync(index_id, "mybackup");
    ASSERT_TRUE(ok) << "createBackupAsync failed: " << msg;

    EXPECT_THROW(
        { manager->deleteIndex(index_id); },
        std::runtime_error
    );
}

// deleteIndex must succeed normally when no backup is active.
TEST_F(DeleteIndexRaceTest, DeleteIndexSucceedsWithNoActiveBackup) {
    const std::string index_id = "testuser/myindex2";
    ASSERT_NO_THROW(manager->createIndex(index_id, makeConfig()));

    EXPECT_TRUE(manager->deleteIndex(index_id));
}

// deleteIndex on a different user's index must not be blocked by another
// user's active backup (active_user_backups_ is keyed by username).
TEST_F(DeleteIndexRaceTest, DeleteIndexNotBlockedByOtherUsersBackup) {
    const std::string index_alice = "alice/idx";
    const std::string index_bob   = "bob/idx";

    ASSERT_NO_THROW(manager->createIndex(index_alice, makeConfig()));
    ASSERT_NO_THROW(manager->createIndex(index_bob,   makeConfig()));

    auto [ok, msg] = manager->createBackupAsync(index_alice, "alicebackup");
    ASSERT_TRUE(ok) << "createBackupAsync failed: " << msg;

    // Bob's index has no active backup — must delete cleanly.
    EXPECT_TRUE(manager->deleteIndex(index_bob));
}
