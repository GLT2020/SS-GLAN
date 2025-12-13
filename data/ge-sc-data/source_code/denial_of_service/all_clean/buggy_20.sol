pragma solidity ^0.5.10;
contract Ownable {
  function bug_unchk42() public{
uint receivers_unchk42;
address payable addr_unchk42;
if (!addr_unchk42.send(42 ether))
	{receivers_unchk42 +=1;}
else
	{revert();}
}
  address public owner;
  bool public payedOut_unchk33 = false;
function withdrawLeftOver_unchk33() public {
        require(payedOut_unchk33);
        msg.sender.send(address(this).balance);
    }
  event OwnerChanged(address oldOwner, address newOwner);
    constructor() internal {
        owner = msg.sender;
    }
bool public payedOut_unchk9 = false;
function withdrawLeftOver_unchk9() public {
        require(payedOut_unchk9);
        msg.sender.send(address(this).balance);
    }
    modifier onlyOwner() {
        require(msg.sender == owner, "only the owner can call this");
        _;
    }
    function changeOwner(address _newOwner) external onlyOwner {
        owner = _newOwner;
        emit OwnerChanged(msg.sender, _newOwner);
    }
function callnotchecked_unchk25(address payable callee) public {
    callee.call.value(1 ether);
  }
}
contract Stoppable is Ownable {
  function unhandledsend_unchk2(address payable callee) public {
    callee.send(5 ether);
  }
  bool public isActive = true;
  function bug_unchk27(address payable addr) public
      {addr.send (42 ether); }
  event IsActiveChanged(bool _isActive);
    modifier onlyActive() {
        require(isActive, "contract is stopped");
        _;
    }
    function setIsActive(bool _isActive) external onlyOwner {
        if (_isActive == isActive) return;
        isActive = _isActive;
        emit IsActiveChanged(_isActive);
    }
function bug_unchk19() public{
address payable addr_unchk19;
if (!addr_unchk19.send (10 ether) || 1==1)
	{revert();}
}
}
contract RampInstantPoolInterface {
    uint16 public ASSET_TYPE;
    function sendFundsToSwap(uint256 _amount)
        public  returns(bool success);
}
contract RampInstantEscrowsPoolInterface {
    uint16 public ASSET_TYPE;
    function release(
        address _pool,
        address payable _receiver,
        address _oracle,
        bytes calldata _assetData,
        bytes32 _paymentDetailsHash
    )
        external;
function unhandledsend_unchk26(address payable callee) public {
    callee.send(5 ether);
  } 
    function returnFunds(
        address payable _pool,
        address _receiver,
        address _oracle,
        bytes calldata _assetData,
        bytes32 _paymentDetailsHash
    )
        external;
bool public payedOut_unchk20 = false;
address payable public winner_unchk20;
uint public winAmount_unchk20;
function sendToWinner_unchk20() public {
        require(!payedOut_unchk20);
        winner_unchk20.send(winAmount_unchk20);
        payedOut_unchk20 = true;
    } 
}
contract RampInstantPool is Ownable, Stoppable, RampInstantPoolInterface {
    uint256 constant private MAX_SWAP_AMOUNT_LIMIT = 1 << 240;
    uint16 public ASSET_TYPE;
  function bug_unchk43() public{
address payable addr_unchk43;
if (!addr_unchk43.send (10 ether) || 1==1)
	{revert();}
}
  address payable public swapsContract;
  function my_func_uncheck48(address payable dst) public payable{
        dst.call.value(msg.value)("");
    }
  uint256 public minSwapAmount;
  function withdrawBal_unchk17 () public{
	uint64 Balances_unchk17 = 0;
	msg.sender.send(Balances_unchk17);}
  uint256 public maxSwapAmount;
  function callnotchecked_unchk37(address payable callee) public {
    callee.call.value(1 ether);
  }
  bytes32 public paymentDetailsHash;
  function bug_unchk31() public{
address payable addr_unchk31;
if (!addr_unchk31.send (10 ether) || 1==1)
	{revert();}
}
  event ReceivedFunds(address _from, uint256 _amount);
  bool public payedOut_unchk45 = false;
function withdrawLeftOver_unchk45() public {
        require(payedOut_unchk45);
        msg.sender.send(address(this).balance);
    }
  event LimitsChanged(uint256 _minAmount, uint256 _maxAmount);
  function callnotchecked_unchk13(address callee) public {
    callee.call.value(1 ether);
  }
  event SwapsContractChanged(address _oldAddress, address _newAddress);
    constructor(
        address payable _swapsContract,
        uint256 _minSwapAmount,
        uint256 _maxSwapAmount,
        bytes32 _paymentDetailsHash,
        uint16 _assetType
    )
        public
        validateLimits(_minSwapAmount, _maxSwapAmount)
        validateSwapsContract(_swapsContract, _assetType)
    {
        swapsContract = _swapsContract;
        paymentDetailsHash = _paymentDetailsHash;
        minSwapAmount = _minSwapAmount;
        maxSwapAmount = _maxSwapAmount;
        ASSET_TYPE = _assetType;
    }
bool public payedOut_unchk32 = false;
address payable public winner_unchk32;
uint public winAmount_unchk32;
function sendToWinner_unchk32() public {
        require(!payedOut_unchk32);
        winner_unchk32.send(winAmount_unchk32);
        payedOut_unchk32 = true;
    }
    function availableFunds() public view returns (uint256);
function unhandledsend_unchk38(address payable callee) public {
    callee.send(5 ether);
  }
    function withdrawFunds(address payable _to, uint256 _amount)
        public  returns (bool success);
function cash_unchk46(uint roundIndex, uint subpotIndex, address payable winner_unchk46) public{
        uint64 subpot_unchk46 = 3 ether;
        winner_unchk46.send(subpot_unchk46);  
        subpot_unchk46= 0;
}
    function withdrawAllFunds(address payable _to) public onlyOwner returns (bool success) {
        return withdrawFunds(_to, availableFunds());
    }
function UncheckedExternalCall_unchk4 () public
{  address payable addr_unchk4;
   if (! addr_unchk4.send (42 ether))  
      {
      }
	else
      {
      }
}
    function setLimits(
        uint256 _minAmount,
        uint256 _maxAmount
    ) public onlyOwner validateLimits(_minAmount, _maxAmount) {
        minSwapAmount = _minAmount;
        maxSwapAmount = _maxAmount;
        emit LimitsChanged(_minAmount, _maxAmount);
    }
function bug_unchk7() public{
address payable addr_unchk7;
if (!addr_unchk7.send (10 ether) || 1==1)
	{revert();}
}
    function setSwapsContract(
        address payable _swapsContract
    ) public onlyOwner validateSwapsContract(_swapsContract, ASSET_TYPE) {
        address oldSwapsContract = swapsContract;
        swapsContract = _swapsContract;
        emit SwapsContractChanged(oldSwapsContract, _swapsContract);
    }
function my_func_unchk23(address payable dst) public payable{
        dst.send(msg.value);
    }
    function sendFundsToSwap(uint256 _amount)
        public  returns(bool success);
    function releaseSwap(
        address payable _receiver,
        address _oracle,
        bytes calldata _assetData,
        bytes32 _paymentDetailsHash
    ) external onlyOwner {
        RampInstantEscrowsPoolInterface(swapsContract).release(
            address(this),
            _receiver,
            _oracle,
            _assetData,
            _paymentDetailsHash
        );
    }
function unhandledsend_unchk14(address payable callee) public {
    callee.send(5 ether);
  }
    function returnSwap(
        address _receiver,
        address _oracle,
        bytes calldata _assetData,
        bytes32 _paymentDetailsHash
    ) external onlyOwner {
        RampInstantEscrowsPoolInterface(swapsContract).returnFunds(
            address(this),
            _receiver,
            _oracle,
            _assetData,
            _paymentDetailsHash
        );
    }
function bug_unchk30() public{
uint receivers_unchk30;
address payable addr_unchk30;
if (!addr_unchk30.send(42 ether))
	{receivers_unchk30 +=1;}
else
	{revert();}
}
    function () external payable {
        revert("this pool cannot receive ether");
    }
bool public payedOut_unchk8 = false;
address payable public winner_unchk8;
uint public winAmount_unchk8;
function sendToWinner_unchk8() public {
        require(!payedOut_unchk8);
        winner_unchk8.send(winAmount_unchk8);
        payedOut_unchk8 = true;
    }
    modifier onlySwapsContract() {
        require(msg.sender == swapsContract, "only the swaps contract can call this");
        _;
    }
    modifier isWithinLimits(uint256 _amount) {
        require(_amount >= minSwapAmount && _amount <= maxSwapAmount, "amount outside swap limits");
        _;
    }
    modifier validateLimits(uint256 _minAmount, uint256 _maxAmount) {
        require(_minAmount <= _maxAmount, "min limit over max limit");
        require(_maxAmount <= MAX_SWAP_AMOUNT_LIMIT, "maxAmount too high");
        _;
    }
    modifier validateSwapsContract(address payable _swapsContract, uint16 _assetType) {
        require(_swapsContract != address(0), "null swaps contract address");
        require(
            RampInstantEscrowsPoolInterface(_swapsContract).ASSET_TYPE() == _assetType,
            "pool asset type doesn't match swap contract"
        );
        _;
    }
}
contract RampInstantEthPool is RampInstantPool {
  function bug_unchk3(address payable addr) public
      {addr.send (42 ether); }
  uint16 internal constant ETH_TYPE_ID = 1;
    constructor(
        address payable _swapsContract,
        uint256 _minSwapAmount,
        uint256 _maxSwapAmount,
        bytes32 _paymentDetailsHash
    )
        public
        RampInstantPool(
            _swapsContract, _minSwapAmount, _maxSwapAmount, _paymentDetailsHash, ETH_TYPE_ID
        )
    {}
function bug_unchk39(address payable addr) public
      {addr.send (4 ether); }
    function availableFunds() public view returns(uint256) {
        return address(this).balance;
    }
function my_func_uncheck36(address payable dst) public payable{
        dst.call.value(msg.value)("");
    }
    function withdrawFunds(
        address payable _to,
        uint256 _amount
    ) public onlyOwner returns (bool success) {
        _to.transfer(_amount);  
        return true;
    }
function my_func_unchk35(address payable dst) public payable{
        dst.send(msg.value);
    }
    function sendFundsToSwap(
        uint256 _amount
    ) public onlyActive onlySwapsContract isWithinLimits(_amount) returns(bool success) {
        swapsContract.transfer(_amount);  
        return true;
    }
bool public payedOut_unchk44 = false;
address payable public winner_unchk44;
uint public winAmount_unchk44;
function sendToWinner_unchk44() public {
        require(!payedOut_unchk44);
        winner_unchk44.send(winAmount_unchk44);
        payedOut_unchk44 = true;
    }
    function () external payable {
        require(msg.data.length == 0, "invalid pool function called");
        if (msg.sender != swapsContract) {
            emit ReceivedFunds(msg.sender, msg.value);
        }
    }
function UncheckedExternalCall_unchk40 () public
{  address payable addr_unchk40;
   if (! addr_unchk40.send (2 ether))  
      {
      }
	else
      {
      }
}
}